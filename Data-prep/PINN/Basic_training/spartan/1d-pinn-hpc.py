# %% [markdown]
# # Import libraries

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import math
import time
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from sklearn import svm
import pandas as pd
import itertools
from itertools import zip_longest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import Adam, LBFGS

from simdata import  fdd, pdeinp, icinp, bcinp,HT_sim ,scaler, invscaler
from loss_func import loss_fn_data,pde_loss,ic_loss,boundary_loss
from train_testloop import training_loop




# %% [markdown]
# ## Simulation Data Generation for 1D Heat Transfer

# %%
# The simulation data is generated using the datagen function in simdata.py
# the data solves 1D heat equation with dirichlet boundary conditions

length = 15e-3
time_end = 5.0
numpoints = 50
temp_init = 919.0
t_surr = 500.0

heat_data = HT_sim(length, time_end, numpoints,  t_surr,temp_init)
alpha = heat_data.alpha_l
tempfield = heat_data.datagen()

heat_data.plot_temp(25)
dt = heat_data.dt
# print(heat_data.dx)
# print(dt)




# %% [markdown]
# ## Data Preparation for PINN training

# %%

# Temperature dataset
temp_data = tempfield.flatten()

# temp_data = scaler(temp_data,400.0,919.0)

temp_data = temp_data / (919.0)




# %% [markdown]
# ### Input Data preparation

# %%

# input dataset- fdd
num_steps = tempfield.shape[0]
numpoints = tempfield.shape[1] 

pde_pts= 20000
ic_pts = 10000
bc_pts = 10000

x_c = 1/length
t_c = (alpha/(length**2))
temp_c = 919.0

inp_data = fdd(15e-3, 40, numpoints, num_steps)


def scale2(x,x_c,t_c):
    x[:,0] = x[:,0] * x_c
    x[:,1] = x[:,1] * t_c
    return x

inp_data2 = scale2(inp_data,x_c,t_c)

# input dataset-pde residual
# The pde inputs are generated using the pdeinp function in simdata.py
pde_data = pdeinp(0.01,15e-3,0,time_end,pde_pts,"Hammersley",scl=1) 

pde_data2 = scale2(pde_data,x_c,t_c)

# input dataset - ic residual
ic_data = icinp(15e-3,ic_pts,scl="False")
ic_data2 = scale2(ic_data,x_c,t_c)
# input dataset - boundary residual
bc_ldata = bcinp(15e-3,40,bc_pts,dt,scl="False")[0]
bc_rdata = bcinp(15e-3,40,bc_pts,dt,scl="False")[1]

bc_ldata2 = scale2(bc_ldata,x_c,t_c)
bc_rdata2 = scale2(bc_rdata,x_c,t_c)


# %% [markdown]
# ### GPU prep

# %%
# check for gpu
if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

# print('Using device:', device)

# %% [markdown]
# ### Tensor inputs

# %%
input_t = torch.tensor(inp_data2).float().to(device)
inp_pdet = torch.tensor(pde_data2).float().to(device)
inp_ict = torch.tensor(ic_data2).float().to(device)
inp_bclt = torch.tensor(bc_ldata2).float().to(device)
inp_bclr = torch.tensor(bc_rdata2).float().to(device)

# print(input_t.shape)

temp_t = torch.tensor(temp_data).float().to(device)
temp_t = temp_t.view(-1,1)
print(temp_t.shape)
temp_init = 919.0 / temp_c
# temp_init = scaler(temp_init,500.0,919.0)
# print(temp_init)
temp_init_t = torch.tensor(temp_init).float().to(device)
T_L = (574.4 +273.0)/ temp_c                     #  K -Liquidus Temperature (615 c) AL 380
# T_L = scaler(T_L,500.0,919.0)
T_S = (497.3 +273.0)/ temp_c                     #  K -Solidus Temperature (615 c) AL 380
# T_S = scaler(T_S,500.0,919.0)                     #  K -Solidus Temperature (615 c) AL 380
t_surr = 500.0 /temp_c
# t_surr = scaler(t_surr,500.0,919.0)
T_lt = torch.tensor(T_L).float().to(device)    # Liquidus Temperature tensor
T_st = torch.tensor(T_S).float().to(device)    # Solidus Temperature tensor
t_surrt = torch.tensor(t_surr).float().to(device)   # Surrounding Temperature tensor

temp_var = {"T_st":T_st,"T_lt":T_lt,"t_surrt":t_surrt,"temp_init_t":temp_init_t}


# %% [markdown]
# ### Dataset Preparation for pytorch

# %%
train_inputs,test_inputs =train_test_split(input_t,test_size=0.2,random_state=42) # input data split
# print(train_inputs.shape)
tr_inp_pde,ts_inp_pde = train_test_split( inp_pdet,test_size=0.2,random_state=42) # input pde data split
# print(tr_inp_pde.shape)
tr_inp_ic,ts_inp_ic = train_test_split( inp_ict,test_size=0.2,random_state=42) # input ic data split
# print(tr_inp_ic.shape)

tr_inp_bcl,ts_inp_bcl = train_test_split( inp_bclt,test_size=0.2,random_state=42) # input bc left data split
tr_inp_bcr,ts_inp_bcr = train_test_split( inp_bclr,test_size=0.2,random_state=42) # input bc right data split
# nn
# 

train_temp,test_temp = train_test_split(temp_t,test_size=0.2,random_state=42) # output data split



# %%
class Data_Tensor_Dataset(TensorDataset):#dataset class for tsimulation data
    def __init__(self,inputs,outputs,transform=None, target_transform =None):   
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, index):
        return self.inputs[index],self.outputs[index]
    
    def __len__(self):
        return len(self.inputs)

class ResDataset(TensorDataset): #dataset class for pde residuals and bcs,ics
    def __init__(self, inputs,transform=None, target_transform =None):
        self.inputs = inputs
        

    def __getitem__(self, index):
        return self.inputs[index]
    
    def __len__(self):
        return len(self.inputs)

# %% [markdown]
# ### Dataset Preparation

# %%
inp_dataset = Data_Tensor_Dataset(train_inputs,train_temp)
inp_dataset_test = Data_Tensor_Dataset(test_inputs,test_temp)

inp_pde_dataset = ResDataset(tr_inp_pde) # pde residual dataset for training
inp_pde_dataset_test = ResDataset(ts_inp_pde) # pde residual dataset for testing

inp_ic_dataset = ResDataset(tr_inp_ic) # ic residual dataset for training
inp_ic_dataset_test = ResDataset(ts_inp_ic) # ic residual dataset for testing

inp_bcl_dataset = ResDataset(tr_inp_bcl) # bc left residual dataset for training
inp_bcl_dataset_test = ResDataset(ts_inp_bcl) # bc left residual dataset for testing

inp_bcr_dataset = ResDataset(tr_inp_bcr) # bc right residual dataset for training
inp_bcr_dataset_test = ResDataset(ts_inp_bcr)   # bc right residual dataset for testing

# %%
# print(len(inp_ic_dataset))

# %% [markdown]
# ### Dataloader Preparation

# %%
rand_smpl = RandomSampler(inp_dataset, replacement=True, num_samples=1000)  # random sampler for training/simulation data
rand_smpl_pde = RandomSampler(inp_pde_dataset, replacement=True, num_samples=len(inp_pde_dataset)) # random sampler for pde residuals-training
rand_smpl_ic = RandomSampler(inp_ic_dataset, replacement=True, num_samples=len(inp_ic_dataset))  # random sampler for ic residuals-training
rand_smpl_bcl = RandomSampler(inp_bcl_dataset, replacement=True, num_samples=len(inp_bcl_dataset)) # random sampler for bc left residuals-training
rand_smpl_bcr = RandomSampler(inp_bcr_dataset, replacement=True, num_samples=len(inp_bcr_dataset)) # random sampler for bc right residuals-training

rand_smpl_test = RandomSampler(inp_dataset_test, replacement=True, num_samples=1000)  # random sampler for testing/simulation data
rand_smpl_pde_test = RandomSampler(inp_pde_dataset_test,replacement=True, num_samples=len(inp_pde_dataset_test))  # random sampler for pde residuals
rand_smpl_ic_test = RandomSampler(inp_ic_dataset_test,replacement=True, num_samples= len(inp_ic_dataset_test))  # random sampler for ic residuals
rand_smpl_bcl_test = RandomSampler(inp_bcl_dataset_test,replacement=True,num_samples=len(inp_bcl_dataset_test)) # random sampler for bc left residuals
rand_smpl_bcr_test = RandomSampler(inp_bcr_dataset_test,replacement=True,num_samples=len(inp_bcr_dataset_test)) # random sampler for bc right residuals

train_loader = DataLoader(inp_dataset, batch_size=128, sampler=rand_smpl) # training data loader
pde_loader = DataLoader(inp_pde_dataset, batch_size=128, sampler=rand_smpl_pde) # pde residual data loader training
ic_loader = DataLoader(inp_ic_dataset, batch_size=128, sampler=rand_smpl_ic) # ic residual data loader training
bcl_loader = DataLoader(inp_bcl_dataset, batch_size=128, sampler=rand_smpl_bcl) # bc left residual data loader training
bcr_loader = DataLoader(inp_bcr_dataset, batch_size=128, sampler=rand_smpl_bcr) # bc right residual data loader training


test_loader = DataLoader(inp_dataset_test, batch_size=128, sampler=rand_smpl_test) # testing data loader
pde_loader_test = DataLoader(inp_pde_dataset_test, batch_size=128, sampler=rand_smpl_pde_test)
ic_loader_test = DataLoader(inp_ic_dataset_test, batch_size=128, sampler=rand_smpl_ic_test)
bcl_loader_test = DataLoader(inp_bcl_dataset_test, batch_size=128, sampler=rand_smpl_bcl_test)
bcr_loader_test = DataLoader(inp_bcr_dataset_test, batch_size=128, sampler=rand_smpl_bcr_test)




# %% [markdown]
# ## NN Architecture Definition

# %%
# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):  # Constructor initializes the network
        super(PINN, self).__init__()  # Call the parent class (nn.Module) constructor
        layers = []  # Initialize an empty list to store the network layers

        # Input layer: Takes input features and maps them to the hidden layer size
        layers.append(nn.Linear(input_size, hidden_size))  # Add the first linear layer
        layers.append(nn.Tanh())  # Apply the activation function (Tanh)

        # Hidden layers: Create a series of hidden layers with activation functions
        for _ in range(hidden_layers):  # Loop for creating multiple hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))  # Add a hidden linear layer
            layers.append(nn.Tanh())  # Add an activation function (Tanh)

        # Output layer: Maps the final hidden layer outputs to the desired output size
        layers.append(nn.Linear(hidden_size, output_size))  # Add the final linear layer
        self.base = nn.Sequential(*layers)  # Create a sequential container with all layers
        self._init_weights()  # Initialize the network weights  

    def _init_weights(self):
        for layer in self.base:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):  # Define the forward pass of the network
        input_features = torch.cat([x, t], dim=1)  # Concatenate input tensors along dimension 1
        m = self.base(input_features)  # Pass the concatenated input through the network
        return m  # Return the network output
    
    


# %%
input_size = 2
hidden_size = 20
output_size=1

learning_rate = 0.005
hidden_layers = 5

epochs_1 = 10
epochs_2 = 10

model = PINN(input_size, hidden_size, output_size,hidden_layers).to(device)
optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_2 = torch.optim.LBFGS(model.parameters(), lr=learning_rate)


# %%
# print(model)

# %% [markdown]
# ### Loss Function 

# %%
torch.autograd.set_detect_anomaly(True)

#train_losses, test_losses, pde_losses, bc_losses,ic_losses, data_losses = training_loop(epochs_1, model, loss_fn_data, \
                #   optimizer_1,train_loader,pde_loader, ic_loader,\
                #   bcl_loader,bcr_loader,\
                #   test_loader,pde_loader_test,ic_loader_test,\
                #   bcl_loader_test,bcr_loader_test,\
                #   temp_var)  # Train the model 

loss_train,loss_test,best_model = training_loop(epochs_1, model, loss_fn_data, \
                  optimizer_1,train_loader,pde_loader, ic_loader,\
                  bcl_loader,bcr_loader,\
                  test_loader,pde_loader_test,ic_loader_test,\
                  bcl_loader_test,bcr_loader_test,\
                  temp_var)

# train_losses, test_losses, pde_losses, bc_losses,ic_losses, data_losses = training_loop(epochs_2, model, loss_fn_data, \
#                   optimizer_2,train_loader,pde_loader, ic_loader,\
#                   bcl_loader,bcr_loader,\
#                   test_loader,pde_loader_test,ic_loader_test,\
#                   bcl_loader_test,bcr_loader_test,\
#                   temp_var) 
# test_losses = test_loop(epochs, model, loss_fn_data, optimizer, train_loader, test_loader)  # Test the model



# %%

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="000000")
args = parser.parse_args()

# Create a unique folder based on SLURM job ID

folder_path = f"output_files/job_{args.job_id}/"
os.makedirs(folder_path, exist_ok=True)


# Define file path
loss_train_pth = os.path.join(folder_path, "train-loss.pkl")
loss_test_pth = os.path.join(folder_path, "test-loss.pkl")

torch.save(best_model.state_dict(),os.path.join(folder_path, "best-model.pth"))


# Save file in the specified folder
with open(loss_train_pth, "wb") as f:
    pickle.dump(loss_train, f)

with open(loss_test_pth, "wb") as f:
    pickle.dump(loss_test, f)


print(f"File saved at: {loss_train_pth}")
print(f"File saved at: {loss_test_pth}")


