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


from loss_func import loss_fn_data,pde_loss,ic_loss,boundary_loss
from train_testloop import training_loop




# %% [markdown]
# ## Simulation Data Generation for 1D Heat Transfer

# %%
# The simulation data is generated using the datagen function in simdata.py
# the data solves 1D heat equation with dirichlet boundary conditions

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)

X, T = np.meshgrid(x, t)

temp = np.zeros((100, 100))


temp[:, 0] = 0
temp[:, -1] = 0
temp[0, :] = 1
a = 1
for i in range(1, 100):
    for j in range(1, 100):
        temp[i, j] = np.exp(-(1**2 * np.pi**2 * a * T[i,j]) / (1**2)) \
            * np.sin(1 * np.pi * X[i,j] / 1)




# %% [markdown]
# ## Data Preparation for PINN training

# %%
# Temperature dataset
temp_data = temp.flatten()



# %% [markdown]
# ### Input Data preparation

# %%

# input dataset- fdd
num_steps = 100
numpoints = 100 

pde_pts= 20000
ic_pts = 1000
bc_pts = 1000

x_c = 1
t_c = 1
temp_c = 1

input_x = X.flatten()
input_t = T.flatten()

input_data = np.stack((input_x,input_t),axis=1)

pde_x = np.linspace(0.1,0.9,pde_pts)
pde_t = np.linspace(0.1,0.9,pde_pts)
pde_data2 = np.stack((pde_x,pde_t),axis=1)

ic_x = np.linspace(0,1,ic_pts)
ic_t = np.zeros(ic_pts)
ic_data = np.stack((ic_x,ic_t),axis=1)

bc_x_l = np.zeros(bc_pts)
bc_t_l = np.linspace(0,1,bc_pts)
bc_data_l = np.stack((bc_x_l,bc_t_l),axis=1)

bc_x_r = np.ones(bc_pts)
bc_t_r = np.linspace(0,1,bc_pts)
bc_data_r = np.stack((bc_x_r,bc_t_r),axis=1)



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

print('Using device:', device)

# %% [markdown]
# ### Tensor inputs

# %%
input_t = torch.tensor(input_data).float().to(device)
inp_pdet = torch.tensor(pde_data2).float().to(device)
inp_ict = torch.tensor(ic_data).float().to(device)
inp_bclt = torch.tensor(bc_data_l).float().to(device)
inp_bclr = torch.tensor(bc_data_r).float().to(device)



temp_t = torch.tensor(temp_data).float().to(device)
temp_t = temp_t.view(-1,1)


# temp_init = scaler(temp_init,500.0,919.0)
temp_init = 1.0
t_surr = 0.0
temp_init_t = 1.0
T_L = (574.4 +273.0)                   #  K -Liquidus Temperature (615 c) AL 380
# T_L_s = scaler(T_L,temp_init, t_surr)                     #  K -Liquidus Temperature (615 c) AL 380
# T_L = scaler(T_L,500.0,919.0)
T_S = (497.3 +273.0)                   #  K -Solidus Temperature (615 c) AL 380
# T_S_s = scaler(T_S,temp_init, t_surr)                     #  K -Solidus Temperature (615 c) AL 380
# T_S = scaler(T_S,500.0,919.0)                     #  K -Solidus Temperature (615 c) AL 380
t_surr_s = 0.0
# t_surr = scaler(t_surr,500.0,919.0)
T_lt = torch.tensor(T_L).float().to(device)    # Liquidus Temperature tensor
T_st = torch.tensor(T_S).float().to(device)    # Solidus Temperature tensor
t_surrt = torch.tensor(t_surr_s).float().to(device)   # Surrounding Temperature tensor

temp_var = {"T_st":T_st,"T_lt":T_lt,"t_surrt":t_surrt,"temp_init_t":temp_init_t}


# %% [markdown]
# ### Dataset Preparation for pytorch

# %%
train_inputs,test_inputs =train_test_split(input_t,test_size=0.2,random_state=42) # input data split
print(train_inputs.shape)
tr_inp_pde,ts_inp_pde = train_test_split( inp_pdet,test_size=0.2,random_state=42) # input pde data split
print(tr_inp_pde.shape)
tr_inp_ic,ts_inp_ic = train_test_split( inp_ict,test_size=0.2,random_state=42) # input ic data split
print(tr_inp_ic.shape)

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
from model import PINN

# %%
input_size = 2
hidden_size = 40
output_size=1

learning_rate = 0.005
hidden_layers = 5

epochs_1 = 30000
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
def move_to_cpu(obj):
    """Recursively move tensors in a dictionary or list to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, list):
        return [move_to_cpu(item) for item in obj]  # Convert tensors inside lists
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}  # Convert tensors inside dicts
    return obj  # Return unchanged for other types

# Ensure all tensors inside lists/dicts are on CPU
loss_train = move_to_cpu(loss_train)
loss_test = move_to_cpu(loss_test)

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
print(f"epoch: {epochs_1}, hidden layers: {hidden_layers}, hidden size: {hidden_size}, learning rate: {learning_rate}")


