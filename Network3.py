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

from sklearn.metrics import mean_absolute_error

#%% Basic parameters

num_samples = 5000
num_div = int(num_samples/4)

num_atoms = 782
num_fasc = 2
num_params = 6
#nombre de paramètres à estimer: ['nu1', 'r1 ', 'f1 ', 'nu2', 'r2 ', 'f2 ']


# %% Data

from getDataW import gen_batch_data

w_store, target_params = gen_batch_data(0, num_samples, 'train')

print(w_store.shape, target_params.shape)
w_reshape = np.zeros((num_samples, num_atoms, num_fasc))

w_reshape[:,:,0] = w_store[:, 0:num_atoms]
w_reshape[:,:,1] = w_store[:, num_atoms: 2*num_atoms]

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

## Dividing in train test and valid

target_train = target_params[0:2*num_div, :]
target_test = target_params[2*num_div:3*num_div, :]
target_valid = target_params[3*num_div:4*num_div, :]

print(target_train[0, :])

print('target_train size', target_train.shape)
print('target_test size', target_test.shape)
print('target_valid size', target_valid.shape)

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
    
    net_tot = Net_tot(num_w_in, num_w_l1, num_w_l2, num_w_out, num_f_in, num_f_l1, num_f_l2, num_f_out)
    
    print(net_tot)
    
    # Optimizer and Criterion
    
    optimizer = optim.Adam(net_tot.parameters(), lr=params["learning_rate"], weight_decay=0.0000001)
    lossf = nn.MSELoss()
    
    
    print('----------------------- Training --------------------------')
    
    # setting hyperparameters and gettings epoch sizes
    batch_size = params["batch_size"] #100 
    num_epochs = params["num_epochs"] #200
    print(x_train.shape)
    shapes = x_train.shape
    num_samples_train = int(num_samples/2)
    num_batches_train = num_samples_train // batch_size #??
    num_samples_valid = int(num_samples/4)
    num_batches_valid = num_samples_valid // batch_size
    
    print(num_batches_train, num_batches_valid)
    
    # setting up lists for handling loss/accuracy
    train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_loss = [], [], [], [], [], [], []
    valid_acc1, valid_acc2, valid_acc3, valid_acc4, valid_acc5, valid_acc6, valid_loss = [], [], [], [], [], [], []
    test_acc, test_loss = [], []
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
        for j in range(num_params):
            train_preds1, train_preds2, train_preds3, train_preds4, train_preds5, train_preds6 = [], [], [], [], [], []
            train_targs1, train_targs2, train_targs3, train_targs4, train_targs5, train_targs6 = [], [], [], [], [], []
            for i in range(num_batches_train):
                slce = get_slice(i, batch_size)
                preds = net_tot(x_train[slce, :, 0], x_train[slce, :, 1])
              
                train_targs1 += list(target_train[slce, 0].numpy())
                train_targs2 += list(target_train[slce, 1].numpy())
                train_targs3 += list(target_train[slce, 2].numpy())
                train_targs4 += list(target_train[slce, 3].numpy())
                train_targs5 += list(target_train[slce, 4].numpy())
                train_targs6 += list(target_train[slce, 5].numpy())
                
                train_preds1 += list(preds.data[:,0].numpy())
                train_preds2 += list(preds.data[:,1].numpy())
                train_preds3 += list(preds.data[:,2].numpy())
                train_preds4 += list(preds.data[:,3].numpy())
                train_preds5 += list(preds.data[:,4].numpy())
                train_preds6 += list(preds.data[:,5].numpy())
            
                
            ### Evaluate validation
            val_preds1, val_preds2, val_preds3, val_preds4, val_preds5, val_preds6 = [], [], [], [], [], []
            val_targs1, val_targs2, val_targs3, val_targs4, val_targs5, val_targs6 = [], [], [], [], [], []
            for i in range(num_batches_valid):
                slce = get_slice(i, batch_size)
                preds = net_tot(x_valid[slce, :, 0], x_valid[slce, :, 1])
                
                val_targs1 += list(target_valid[slce, 0].numpy())
                val_targs2 += list(target_valid[slce, 1].numpy())
                val_targs3 += list(target_valid[slce, 2].numpy())
                val_targs4 += list(target_valid[slce, 3].numpy())
                val_targs5 += list(target_valid[slce, 4].numpy())
                val_targs6 += list(target_valid[slce, 5].numpy())
                
                val_preds1 += list(preds.data[:,0].numpy())
                val_preds2 += list(preds.data[:,1].numpy())
                val_preds3 += list(preds.data[:,2].numpy())
                val_preds4 += list(preds.data[:,3].numpy())
                val_preds5 += list(preds.data[:,4].numpy())
                val_preds6 += list(preds.data[:,5].numpy())
                
        
            train_acc_cur1 = mean_absolute_error(train_targs1, train_preds1)
            train_acc_cur2 = mean_absolute_error(train_targs2, train_preds2)
            train_acc_cur3 = mean_absolute_error(train_targs3, train_preds3)
            train_acc_cur4 = mean_absolute_error(train_targs4, train_preds4)
            train_acc_cur5 = mean_absolute_error(train_targs5, train_preds5)
            train_acc_cur6 = mean_absolute_error(train_targs6, train_preds6)
            
            #print(len(val_targs1), len(val_preds1))
            valid_acc_cur1 = mean_absolute_error(val_targs1, val_preds1)
            valid_acc_cur2 = mean_absolute_error(val_targs2, val_preds2)
            valid_acc_cur3 = mean_absolute_error(val_targs3, val_preds3)
            valid_acc_cur4 = mean_absolute_error(val_targs4, val_preds4)
            valid_acc_cur5 = mean_absolute_error(val_targs5, val_preds5)
            valid_acc_cur6 = mean_absolute_error(val_targs6, val_preds6)
            
            train_acc1.append(train_acc_cur1)
            valid_acc1.append(valid_acc_cur1)
            train_acc2.append(train_acc_cur2)
            valid_acc2.append(valid_acc_cur2)
            train_acc3.append(train_acc_cur3)
            valid_acc3.append(valid_acc_cur3)
            train_acc4.append(train_acc_cur4)
            valid_acc4.append(valid_acc_cur4)
            train_acc5.append(train_acc_cur5)
            valid_acc5.append(valid_acc_cur5)
            train_acc6.append(train_acc_cur6)
            valid_acc6.append(valid_acc_cur6)
        
        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc 1 %f, Valid acc 1 %f, Train acc 2 %f, Valid acc 2 %f" % (
                    epoch+1, losses[-1], train_acc_cur1, valid_acc_cur1, train_acc_cur2, valid_acc_cur2))
        
    
    # epoch = np.arange(len(train_acc1))
    
    # print('--- N_w ---')
    # print('num_w_l1', num_w_l1, '\n', 'num_w_l2', num_w_l2, '\n', 'num_w_out', num_w_out, '\n')
    
    # print('--- N_f ---')
    # print('num_f_l1', num_f_l1, '\n', 'num_f_l2', num_f_l2, '\n')
    
    # plt.figure()
    # plt.plot(epoch, train_acc1, 'r', epoch, valid_acc1, 'b')
    # plt.legend(['Train Accucary','Validation Accuracy'])
    # plt.xlabel('Updates'), plt.ylabel('Acc')
    # plt.show()
    
    # plt.figure()
    # plt.plot(epoch, train_acc2, 'r', epoch, valid_acc2, 'b')
    # plt.legend(['Train Accucary','Validation Accuracy'])
    # plt.xlabel('Updates'), plt.ylabel('Acc')    
    # plt.show()
    
    # plt.figure()
    # plt.plot(epoch, train_acc3, 'r', epoch, valid_acc3, 'b')
    # plt.legend(['Train Accucary','Validation Accuracy'])
    # plt.xlabel('Updates'), plt.ylabel('Acc')
    # plt.show()
    
    # plt.figure()
    # plt.plot(epoch, train_acc4, 'r', epoch, valid_acc4, 'b')
    # plt.legend(['Train Accucary','Validation Accuracy'])
    # plt.xlabel('Updates'), plt.ylabel('Acc')
    # plt.show()
    
    # plt.figure()
    # plt.plot(epoch, train_acc5, 'r', epoch, valid_acc5, 'b')
    # plt.legend(['Train Accucary','Validation Accuracy'])
    # plt.xlabel('Updates'), plt.ylabel('Acc')
    # plt.show()
    
    # plt.figure()
    # plt.plot(epoch, train_acc6, 'r', epoch, valid_acc6, 'b')
    # plt.legend(['Train Accucary','Validation Accuracy'])
    # plt.xlabel('Updates'), plt.ylabel('Acc')
    # plt.show()
    
    
    return {"loss": -valid_acc_cur6, 
            "model": net_tot, 
            "params": params, 
            "status": STATUS_OK,
            "train_acc5":valid_acc_cur6}
    

train_network(params)
