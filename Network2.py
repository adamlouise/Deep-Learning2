#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:23:10 2020

@author: louiseadam
"""

#import matplotlib
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import glob
import os
import sys
import pickle
import time

#from IPython.display import clear_output
#from skimage.io import imread
#from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.parameter import Parameter

from sklearn.preprocessing import StandardScaler

path_to_utils = os.path.join('.', 'python_functions')
path_to_utils = os.path.abspath(path_to_utils)
if path_to_utils not in sys.path:
    sys.path.insert(0, path_to_utils)

import mf_utils as util
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from scipy import stats

from sklearn.metrics import mean_absolute_error

#%% Basic parameters

num_atoms = 782
num_fasc = 2
num_params = 6 #nombre de paramètres à estimer: ['nu1', 'r1 ', 'f1 ', 'nu2', 'r2 ', 'f2 ']


params = {
    #Training parameters
    "num_samples": 1000000,
     "batch_size": 500,  
     "num_epochs": 50,
     
     #NW1 parameters
     #"num_w_out": hp.choice("num_w_out", [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30] ),
     "num_w_out": 12,
     "num_w_l1": 500,
     "num_w_l2": 50,
     
     #NW2
     "num_f_l1": 200,
     "num_f_l2": 100,
     
     #other
     "learning_rate": 0.001,
     #"learning_rate": hp.uniform("learningrate", 0.0005, 0.01),
     "dropout": 0.3
     #"dropout": hp.uniform("dropout", 0, 0.4)
     #hp.choice(hsjdkfhs, )
}

num_samples = params["num_samples"]
num_div = int(num_samples/4)

# %% Data

from getDataW import gen_batch_data

#w_store1, target_params1 = gen_batch_data(0, num_div*2, 'train')
#w_store2, target_params2 = gen_batch_data(0, num_div*2, 'validation')

w_store, target_params = gen_batch_data(0, num_div*4, 'train')
w_store1 = w_store[0:num_div*2, :]
w_store2 = w_store[num_div*2:num_div*4, :]
target_params1 = target_params[0:num_div*2, :]
target_params2 = target_params[num_div*2:num_div*4, :]

print(target_params1[0, :])

w_reshape = np.zeros((num_samples, num_atoms, num_fasc))

w_re = np.zeros((num_samples, ))
w_reshape[0:num_div*2, :,0] = w_store1[:, 0:num_atoms]
w_reshape[0:num_div*2,:,1] = w_store1[:, num_atoms: 2*num_atoms]
w_reshape[num_div*2:num_samples,:,0] = w_store2[:, 0:num_atoms]
w_reshape[num_div*2:num_samples,:,1] = w_store2[:, num_atoms: 2*num_atoms]

print(w_reshape.shape)


# %% Train data

print('----------------------- Data --------------------------')

# divide data in train, test and validation
x_train = w_reshape[0:2*num_div, :, :]
x_test = w_reshape[2*num_div : 3*num_div , :, :]
x_valid = w_reshape[3*num_div : 4*num_div, :, :]

print('x_train size', x_train.shape)
print('x_test size', x_test.shape)
print('x_valid size', x_valid.shape)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
x_valid = torch.from_numpy(x_valid)

# quelques modifs pour le modele neuronal
x_train = x_train.float()
x_test = x_test.float()
x_valid = x_valid.float()


# %% Target data

print("--- Taking microstructural properties of fascicles ---")

#Scaling: scaler: (num_samples, num_features)

scaler_train = StandardScaler()
target_params1 = scaler_train.fit_transform(target_params1)
target_params1 = torch.from_numpy(target_params1)

scaler_valid = StandardScaler()
target_params2 = scaler_valid.fit_transform(target_params2)
target_params2 = torch.from_numpy(target_params2)

## Dividing in train test and valid

target_train = target_params1[:, :]
target_test = target_params2[0:num_div, :]
target_valid = target_params2[num_div:2*num_div, :]

target_train = target_train.float()
target_test = target_test.float()
target_valid = target_valid.float()

print(target_train[0, :])

print('target_train size', target_train.shape)
print('target_test size', target_test.shape)
print('target_valid size', target_valid.shape)



# %% Defining the networks

# Network 1

class Net_w(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_out, drop_prob):
        super(Net_w, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)

        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_3, self.b_3)

        return x


# Network 2
class Net_f(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_out, drop_prob):
        super(Net_f, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.linear(x, self.W_3, self.b_3)

        return x

# Network 3
class Net_tot(nn.Module):

    def __init__(self, numw_in, numw_l1, numw_l2, numw_out, numf_in, numf_l1, numf_l2, numf_out, drop):
        super(Net_tot, self).__init__()  
        self.netw = Net_w(numw_in, numw_l1, numw_l2, numw_out, drop_prob=drop)
        self.netf = Net_f(numf_in, numf_l1, numf_l2, numf_out, drop_prob=drop)

    def forward(self, w1, w2):
        x1 = self.netw(w1)
        x2 = self.netw(w2)
        
        x = torch.cat((x1, x2), axis=1)

        x = self.netf(x)

        return x

#%% 

def train_network(params: dict):
    # Building training loop
    num_w_out = params["num_w_out"] #??
    num_w_l1 = params["num_w_l1"]
    num_w_l2 = params["num_w_l2"]
    num_w_in = num_atoms
    num_f_out = num_params #nombre de paramètres à estimer
    num_f_l1 = params["num_f_l1"]
    num_f_l2 = params["num_f_l2"]
    num_f_in = num_w_out*num_fasc #ici 10*2
    drop = params["dropout"]
    
    net_tot = Net_tot(num_w_in, num_w_l1, num_w_l2, num_w_out, num_f_in, num_f_l1, num_f_l2, num_f_out, drop)
    
    print(net_tot)
    
    # Optimizer and Criterion
    
    optimizer = optim.Adam(net_tot.parameters(), lr=params["learning_rate"], weight_decay=0.0000001)
    lossf = nn.MSELoss()
    
    
    print('----------------------- Training --------------------------')
    
    start = time.time()
    
    # setting hyperparameters and gettings epoch sizes
    batch_size = params["batch_size"] #100 
    num_epochs = params["num_epochs"] #200
    #print(x_train.shape)
    
    #shapes = x_train.shape
    num_samples_train = int(num_samples/2)
    num_batches_train = num_samples_train // batch_size #??
    num_samples_valid = int(num_samples/4)
    num_batches_valid = num_samples_valid // batch_size
    
    print(num_batches_train, num_batches_valid)
    
    # setting up lists for handling loss/accuracy
    train_acc = np.zeros((num_epochs, num_params))
    valid_acc = np.zeros((num_epochs, num_params))
    
    meanTrainError, meanValError  = [], []

    cur_loss = 0
    losses = []
    
    # lambda function
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        ## Train
        cur_loss = 0
        net_tot.train()
        
        for i in range(num_batches_train):
            
            optimizer.zero_grad()
            slce = get_slice(i, batch_size)
    
            output = net_tot(x_train[slce, :, 0], x_train[slce, :, 1])
            
            # compute gradients given loss
            target_batch = target_train[slce]
            batch_loss = lossf(output, target_batch)
            batch_loss.backward()
            optimizer.step()
            
            cur_loss += batch_loss   
        losses.append(cur_loss / batch_size)
        #print(cur_loss / batch_size)
    
        net_tot.eval()
        
        ### Evaluate training
        train_preds = [[], [], [], [], [], []]
        train_targs = [[], [], [], [], [], []]
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            preds = net_tot(x_train[slce, :, 0], x_train[slce, :, 1])
            
            for j in range(num_params):
                train_targs[j] += list(target_train[slce, j].numpy())
                train_preds[j] += list(preds.data[:,j].numpy())
            
        ### Evaluate validation
        val_preds = [[], [], [], [], [], []]
        val_targs = [[], [], [], [], [], []]
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            preds = net_tot(x_valid[slce, :, 0], x_valid[slce, :, 1])
            
            for j in range(num_params):
                val_targs[j] += list(target_valid[slce, j].numpy())
                val_preds[j] += list(preds.data[:,j].numpy())
                
        # Save evaluation and training
        train_acc_cur = np.zeros(num_params)
        valid_acc_cur = np.zeros(num_params)
        for j in range(num_params):
            train_acc_cur[j] = mean_absolute_error(train_targs[j], train_preds[j])
            valid_acc_cur[j] = mean_absolute_error(val_targs[j], val_preds[j])
            train_acc[epoch, j] = train_acc_cur[j]
            valid_acc[epoch, j] = valid_acc_cur[j]
        
        #print(train_acc)
        #print(train_acc)
        meanTrainError.append(np.mean(train_acc[epoch,:]))
        meanValError.append(np.mean(valid_acc[epoch, :]))
        
        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f, " %(
                    epoch+1, losses[-1], meanTrainError[-1], meanValError[-1]))
        
    to_min = sum(valid_acc_cur)
    
    end = time.time()
    t = end-start
      
    return {"loss": to_min, 
            "model": net_tot, 
            "params": params, 
            "status": STATUS_OK,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "meanTrainError": meanTrainError,
            "meanValError": meanValError,
            "time": t
            }
    
#train_network(params)


#%%Testing Optimisation

#trials = Trials()
#best = fmin(train_network, params, algo=tpe.suggest, max_evals=1, trials=trials)

#print(trials.best_trial['result']['loss'])

#train_acc = trials.best_trial['result']['train_acc']


#%% Graph of dropout rate

# # Save Network and load again

# filename = 'network4' 
# with open(filename, 'wb') as f:
#         pickle.dump(trials, f)
#         f.close()

# #trials = pickle.load(open(filename, 'rb'))

# n = len(trials.results)
# tomin = np.zeros(n)
# to_opti = np.zeros(n)
# for i in range(n):
#     tomin[i]= trials.results[i]['loss']
#     to_opti[i] = trials.results[i]['params']['num_w_out']
    
# print(tomin)
# print(to_opti)

# #plt.figure()
# #plt.plot(dropout, tomin, 'b')
# #plt.title('Optimisation of dropout (lr=0.001)')
# #plt.grid(b=True, which='major', color='#666666', linestyle='-')
# #plt.minorticks_on()
# #plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# #plt.xlabel('dropout'), plt.ylabel('Sum of errors')
# #plt.show()

# plt.figure()
# plt.scatter(to_opti, tomin)
# plt.title('Influence of number of output of split network (dropout=0.05, lr=0.001)')
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.xlabel('nun_w_out'), plt.ylabel('Sum of errors')
# plt.show()



#%%

trial = train_network(params)

#train_acc = trials.best_trial['result']['train_acc']
#valid_acc = trials.best_trial['result']['valid_acc']
train_acc = trial['train_acc']
valid_acc = trial['valid_acc']
epoch = np.arange(params['num_epochs'])

mean_train_error = trial['meanTrainError']

for j in range(num_params):    
    plt.figure()
    plt.plot(epoch, train_acc[:, j], 'r', epoch, valid_acc[:, j], 'b')
    plt.legend(['Train error','Validation error'])
    plt.xlabel('Updates'), plt.ylabel('Error')
    plt.show()
    
meanTrainError = trial['meanTrainError']
meanValError = trial['meanValError']

# Mean Error
print(trial['meanTrainError'])
plt.figure()
plt.plot(epoch, meanTrainError, 'r', epoch, meanValError, 'b')
plt.title('Mean Error')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(['Mean Train error','Mean Validation error'])
plt.xlabel('Updates'), plt.ylabel('Error')
plt.show()



#%% Comments
#print("Last accuracies for 100 000 samples for uniform", valid_acc_cur1, valid_acc_cur2, valid_acc_cur3, valid_acc_cur4, valid_acc_cur5, valid_acc_cur6)

## %% Results
#
print('----------------------- Prediction --------------------------')
#
##making prediction
#
net_tot = trial['model']

print(x_test.shape)
output = net_tot(x_test[:,:,0], x_test[:,:,1])
output = output.detach().numpy()


output = scaler_valid.inverse_transform(output)
target_test = scaler_valid.inverse_transform(target_test)

print(target_test[0, :])

# print(output.shape)
# print(target_test.shape)
# print(output[:5])
# print(target_test[:5])
error = output - target_test

abserror = abs(error)
#
##plt.figure()
##plt.plot(range(len(target_test)), error)
##plt.xlabel('samples')
##plt.ylabel('Abs error')
##plt.show()
#
#plt.figure()
#plt.hist(abserror[:,0], density=False, bins=30)  # `density=False` would make counts
#plt.ylabel('Probability')
#plt.xlabel('Data')
#plt.show()
#
#

#%% obtenir 95% interval

conf_int = np.zeros(num_params)

for j in range(num_params):
    #plt.plot(error[:,j])
    #plt.show()
    data = error[:,j]
    
    mean = np.mean(data)
    sigma = np.std(data)
    
    confint = stats.norm.interval(0.95, loc=mean, 
        scale=sigma)
    
    print(confint)


    
#
##%% Save Network and load again
#
#filename = 'network2' 
#with open(filename, 'wb') as f:
#        pickle.dump(net_tot, f)
#        f.close()
#
#loaded_network = pickle.load(open(filename, 'rb'))
#
#print(loaded_network)





