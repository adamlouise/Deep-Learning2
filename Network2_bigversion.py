#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:23:10 2020

@author: louiseadam
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.parameter import Parameter

from sklearn.preprocessing import StandardScaler

#path_to_utils = os.path.join('.', 'python_functions')
#path_to_utils = os.path.abspath(path_to_utils)
#if path_to_utils not in sys.path:
#    sys.path.insert(0, path_to_utils)

#import mf_utils as util
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from scipy import stats

from sklearn.metrics import mean_absolute_error

#%% Basic parameters

num_atoms = 782
num_fasc = 2
num_params = 6 #nombre de paramètres à estimer: ['nu1', 'r1 ', 'f1 ', 'nu2', 'r2 ', 'f2 ']
new_gen = False
nouvel_enregist = False
via_pickle = True

params = {
    #Training parameters
    "num_samples": 1000000,
     "batch_size": 10000,  
     "num_epochs": 30,
     
     #NW1 parameters
     #"num_w_out": hp.choice("num_w_out", [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30] ),
     "num_w_out": 50,
     "num_w_l1": 200,
     "num_w_l2": 600,
     "num_w_l3": 200,
     "num_w_l4": 50,
     
     #NW2
     "num_f_l1": 300,
     "num_f_l2": 200,
     "num_f_l3": 100,
     
     #other
     "learning_rate": 0.002, 
     #"learning_rate": hp.uniform("learningrate", 0.0005, 0.01),
     "dropout": 0.05
     #"dropout": hp.uniform("dropout", 0, 0.4)
}

num_samples = params["num_samples"]
num_div = int(num_samples/4)

print('OK')

# %% Data via pickle files

print('Aller c est partii')
filename1 = 'dataNW2_version1/dataNW2_w_store_version1'
filename2 = 'dataNW2_version1/dataNW2_targets_version1' 

if new_gen:
    
    print("on load avec gen_batch_data")
    
    from getDataW import gen_batch_data

    #w_store1, target_params1 = gen_batch_data(0, num_div*2, 'train')
    #w_store2, target_params2 = gen_batch_data(0, num_div*2, 'validation')

    w_store, target_params = gen_batch_data(0, num_div*4, 'train')
    print(w_store.shape, target_params.shape)
    
    if nouvel_enregist:
        print('et on enregistre :-) ')
        with open(filename1, 'wb') as f:
                pickle.dump(w_store, f)
                f.close()
        with open(filename2, 'wb') as f:
                pickle.dump(target_params, f)
                f.close()

if via_pickle:   
    print("on load via les fichiers pickle :-) ")     
    w_store = pickle.load(open(filename1, 'rb'))
    target_params = pickle.load(open(filename2, 'rb'))    

#%%
w_store1 = w_store[0:num_div*2, :]
w_store2 = w_store[num_div*2:num_div*4, :]
target_params1 = target_params[0:num_div*2, :]
target_params2 = target_params[num_div*2:num_div*4, :]

print(target_params1[0, :])

w_reshape = np.zeros((num_samples, num_atoms, num_fasc))

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

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_out, drop_prob):
        super(Net_w, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h3, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_h3), 0))
        
        self.W_4 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
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
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_4, self.b_4)

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

    def __init__(self, numw_in, numw_l1, numw_l2, numw_l3, numw_out, numf_in, numf_l1, numf_l2, numf_out, drop):
        super(Net_tot, self).__init__()  
        self.netw = Net_w(numw_in, numw_l1, numw_l2, numw_l3, numw_out, drop_prob=drop)
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
    num_w_out = params["num_w_out"] 
    num_w_l1 = params["num_w_l1"]
    num_w_l2 = params["num_w_l2"]
    num_w_l3 = params["num_w_l3"]
    num_w_in = num_atoms
    num_f_out = num_params #nombre de paramètres à estimer
    num_f_l1 = params["num_f_l1"]
    num_f_l2 = params["num_f_l2"]
    num_f_in = num_w_out*num_fasc #ici 10*2
    drop = params["dropout"]
    
    net_tot = Net_tot(num_w_in, num_w_l1, num_w_l2, num_w_l3, num_w_out, num_f_in, num_f_l1, num_f_l2, num_f_out, drop)
    
    print(net_tot)
    
    # Optimizer and Criterion
    
    #optimizer = optim.Adam(net_tot.parameters(), lr=params["learning_rate"], weight_decay=0.0000001)
    optimizer = optim.Adam(net_tot.parameters(), lr=params["learning_rate"])
    lossf = nn.MSELoss()
    
    
    print('----------------------- Training --------------------------')
    
    start = time.time()
    
    # setting hyperparameters and gettings epoch sizes
    batch_size = params["batch_size"] #100 
    num_epochs = params["num_epochs"] #200
    
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
        
        meanTrainError.append(np.mean(train_acc[epoch,:]))
        meanValError.append(np.mean(valid_acc[epoch, :]))
        
        #if epoch % 10 == 0:
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

#%% Training the network 

tic = time.time()

trial = train_network(params)

toc = time.time()
train_time = toc - tic
print("training time: ", train_time)

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

# Mean Error --> Learning curve
print(trial['meanTrainError'])
plt.figure()
plt.plot(epoch, meanTrainError, 'r', epoch, meanValError, 'b')
plt.title('Learning Curve: Mean Error - DL after NNLS')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(['Mean Train error','Mean Validation error'])
plt.xlabel('Updates'), plt.ylabel('Error')
plt.show()


# %% Predictions (Testing)
print('----------------------- Prediction --------------------------')

net_tot = trial['model']

# predict and time
tic = time.time()
output = net_tot(x_test[:,:,0], x_test[:,:,1])
output = output.detach().numpy()
toc = time.time()
predic_time = toc - tic
print("prediction time: ", predic_time)

# mean absolute scaled error for 6 properties
mean_err_scaled = np.zeros(6)
for i in range(6):
    mean_err_scaled[i] = mean_absolute_error(output[:,i], target_test[:,i])

print("mean_abs_err", mean_err_scaled)

properties = ['nu1', 'r1', 'f1', 'nu2', 'r2', 'f2']
plt.figure()
plt.bar(properties, mean_err_scaled)

# descale
output = scaler_valid.inverse_transform(output)
target_test = scaler_valid.inverse_transform(target_test)

error = output - target_test

abserror = abs(error)

plt.figure()
plt.hist(abserror[:,0], density=False, bins=30)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()

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


#%%Testing Optimisation: for this the params dictionnary needs to be changed (at the beginning of the code)

# trials = Trials()
# best = fmin(train_network, params, algo=tpe.suggest, max_evals=1, trials=trials)

# print(trials.best_trial['result']['loss'])

# train_acc = trials.best_trial['result']['train_acc']

# ## GRAPHS FOR OPTIMISATION

# # Save Network and load again

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