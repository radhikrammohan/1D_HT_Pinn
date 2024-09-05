# %% [markdown]
# # 1D Three Phase Simulation of Alloys and PINN model development 
# 

# %% [markdown]
# This notebook contains the simulation of 1D Phase change of aluminium alloy. There will be three phases (solid,liquid and mushy).   
# 
# The approach used is finite difference method and the physics involved in heat conduction.

# %% [markdown]
# ## Import Libraries

# %%
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from sklearn import svm
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from pinn_loss import loss_fn_data, l1_regularization, pde_loss, boundary_loss, ic_loss, accuracy
from Input_vec_gen import input_gen


# %% [markdown]
# ## Simulation

# %% [markdown]
# ### Define the constants and inital geometric domain

# %% [markdown]
# 
# 
# Material :- AL 380
# 
# | Sr.No | Properties  | Symbol | Value  | Unit |Range(source) |
# |:---:|:---:|:---:|:---:|:---:|:--:|
# | 1  | Liquidus Density | $\rho_{l}$  | 2300 | $kg/m^3$  |  2200-2400 (ASM handbook) |
# |  2 |  Solidus Density  |  $\rho_{s}$  | 2500  |  $kg/m^3$  | 2400-2600 (ASM handbook) |
# |  3 |  Mushy Desnity |  $\rho_{m}$  |  2400 | $kg/m^3$   |Increase linearly from liquid to solid (ASM handbook) |
# |  4 |  Liquidus Thermal Conductivity| $k_l$  |  70 | $W/m-K$  |60-80 (ASM handbook) |
# |  5 |  Solidus Thermal Conductivity | $k_s$  | 180  |  $W/m-K$ | 150-210(ASM handbook) |
# |  6 | Mushy Zone Thermal Conductivity | $k_m$  | 125  |  $W/m-K$ |Decrease linearly from liquid to solid (ASM handbook) |
# |  7 | Liquidus Specific Heat | $c_{pl}$  | 1190  | $J/kg-K$  | 1100 -1200 (ASM handbook)|
# |  8 | Solidus Specific Heat | $c_{ps}$  |  1100 |  $J/kg-K$  | 1100-1200 (ASM handbook)|
# |  9 | Mushy Zone Specific Heat |  $c_{pm}$ | 1175 | $J/kg-K$   |decrease lineraly from liquid to solid (ASM handbook)|
# |  10 | Latent Heat of Fusion | $L_{fusion}$  | 450e3  | $J/kg$ | (400-500)e3 (ASM handbook) |
# | 11 | Left Boundary Temperature |$BCT_{l}$|623 |$K$| (623-723) Nissan Data |
# |12 | Right Boundary Temperature | $BCT_{r}$|623 |$K$| (623-723) Nissan Data |
# |13| Freezing time | |60 |sec|||

# %%
# Geometry
length = 15.0e-3             # Length of the rod

# Material properties
rho = 2300.0                     # Density of AL380 (kg/m^3)
rho_l = 2460.0                   # Density of AL380 (kg/m^3)
rho_s = 2710.0                    # Density of AL380 (kg/m^3)
rho_m = (rho_l + rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

k = 104.0                       # W/m-K
k_l = k                       # W/m-K
k_s = 96.2                    # W/m-K
k_m =  (k_l+k_s)/2                     # W/m-K
k_mo = 41.5


cp = 1245.3                      # Specific heat of aluminum (J/kg-K)
cp_l = cp                      # Specific heat of aluminum (J/kg-K)
cp_s = 963.0                 # Specific heat of aluminum (J/kg-K)
cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity
alpha_l = k_l / (rho_l * cp_l) 
alpha_s = k_s / (rho_s*cp_s)
alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`


#L_fusion = 3.9e3                 # J/kg
L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum
         # Thermal diffusivity


T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380
T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)
m_eff =(k_m/(rho_m*(cp_m + (L_fusion/(T_L-T_S)))))
print (f"alpha_l = {alpha_l}, alpha_s = {alpha_s}, m_eff = {m_eff}")

# htc = 10.0                   # W/m^2-K
# q = htc*(919.0-723.0)
# q = 10000.0


num_points = 50                        # Number of spatial points
dx = length / (num_points - 1)         # Distance between two spatial points
print('dx is',dx)

                                                              
# Time Discretization  
# 
time_end = 40        # seconds                         

maxi = max(alpha_s,alpha_l,alpha_m)
dt = abs(0.5*((dx**2) /maxi)) 

print('dt is ',dt)
num_steps = round(time_end/dt)
print('num_steps is',num_steps)
cfl = 0.5 *(dx**2/max(alpha_l,alpha_s,alpha_m))
print('cfl is',cfl)

time_steps = np.linspace(0, time_end, num_steps + 1)
step_coeff = dt / (dx ** 2)

if dt <= cfl:
    print('stability criteria satisfied')
else:
    print('stability criteria not satisfied')
    sys.exit()

# %% [markdown]
# ### Initial and Boundary Conditions

# %%

temp_init = 919.0
# Initial temperature and phase fields
temperature = np.full(num_points+2, 919.0)            # Initial temperature of the rod with ghost points at both ends
phase = np.zeros(num_points+2)*0.0                    # Initial phase of the rod with ghost points at both ends

# Set boundary conditions
# temperature[-1] = 919.0 
phase[-1] = 1.0

# temperature[0] = 919.0 #(40 C)
phase[0] = 1.0

# Store initial state in history
temperature_history = [temperature.copy()]    # List to store temperature at each time step
phi_history = [phase.copy()]                    # List to store phase at each time step
temp_init = temperature.copy()                 # Initial temperature of the rod
# print(temperature_history,phi_history)
# Array to store temperature at midpoint over time
midpoint_index = num_points // 2                          # Index of the midpoint

midpoint_temperature_history = [temperature[midpoint_index]]            # List to store temperature at midpoint over time
dm = 60.0e-3                                                            # die thickness in m

# r_m =  (k_mo / dm) + (1/htc)

t_surr = 500.0                                        # Surrounding temperature in K
# t_surr = h()

def kramp(temp,v1,v2,T_L,T_s):                                      # Function to calculate thermal conductivity in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        if temp > T_L:
            k_m = k_l
        elif temp < T_S:
            k_m = k_s
        else:
            k_m = k_s + slope*(temp-T_S)
        return k_m

def cp_ramp(temp,v1,v2,T_L,T_s):                                    # Function to calculate specific heat capacity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    if temp > T_L:
        cp_m = cp_l
    elif temp < T_S:
        cp_m = cp_s
    else:
        cp_m = cp_s + slope*(temp-T_S)
    return cp_m

def rho_ramp(temp,v1,v2,T_L,T_s):                                       # Function to calculate density in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    if temp > T_L:
        rho_m = rho_l
    elif temp < T_S:
        rho_m = rho_s
    else:
        rho_m = rho_s + slope*(temp-T_S)
    return rho_m

# %% [markdown]
# ### Solving the HT equation and phase change numerically

# %%

for m in range(1, num_steps+1):                                                                            # time loop
    htc = 10.0                   # htc of Still air in W/m^2-K
    q1 = htc*(temp_init[0]-t_surr)   # Heat flux at the left boundary
    
    # print(f"q1 is {q1}")
    temperature[0] = temp_init[0] + alpha_l * step_coeff * ((2.0*temp_init[1]) - (2.0 * temp_init[0])-(2.0*dx*(q1)))  # Update boundary condition temperature
    
    q2 = htc*(temp_init[-1]-t_surr)                   # Heat flux at the right boundary
    temperature[-1] = temp_init[-1] + alpha_l * step_coeff * ((2.0*temp_init[-2]) - (2.0 * temp_init[-1])-(2.0*dx*(q2)))  # Update boundary condition temperature
    
    for n in range(1,num_points+1):              # space loop, adjusted range
       
        if temperature[n] >= T_L:
            temperature[n] += ((alpha_l * step_coeff) * (temp_init[n+1] - (2.0 * temp_init[n]) + temp_init[n-1]))
            phase[n] = 0
            
            # print(f" Time-Step{m},Spatial point{n},Temperature{temperature[n]}")
        elif T_S < temperature[n] < T_L:
            
            k_m = kramp(temperature[n],k_l,k_s,T_L,T_S)
            cp_m = cp_ramp(temperature[n],cp_l,cp_s,T_L,T_S)
            rho_m = rho_ramp(temperature[n],rho_l,rho_s,T_L,T_S)
            m_eff =(k_m/(rho_m*(cp_m + (L_fusion/(T_L-T_S)))))
            
            temperature[n] += ((m_eff * step_coeff)* (temp_init[n+1] - (2.0 * temp_init[n]) + temp_init[n-1]))
            
            phase[n] = (T_L - temperature[n]) / (T_L - T_S)
            # print(m,n,temperature[n],phase[n])
         
        elif temperature[n]<T_S:
            temperature[n] += ((alpha_s * step_coeff) * (temp_init[n+1] - (2.0 * temp_init[n])+ temp_init[n-1]))
            phase[n] = 1
                     
        else:
            print("ERROR: should not be here")

     
          
    temperature = temperature.copy()                                                                # Update temperature
    phase = phase.copy()                                                                            # Update phase
    temp_init = temperature.copy()                                                                  # Update last time step temperature
    temperature_history.append(temperature.copy())                                                  # Append the temperature history to add ghost points
    phi_history.append(phase.copy())                                                                # Append the phase history to add ghost points
    midpoint_temperature_history.append(temperature[midpoint_index])                                # Store midpoint temperature
    
    
    # print(f"Step {m}, Temperature: {temperature}")
    


# print(midpoint_temperature_history)
#print(phi_history)





# %% [markdown]
# ### Plot the Results

# %%
# # Plot temperature history for debugging
# temperature_history_1 = np.array(temperature_history)
# print(temperature_history_1.shape)
# time_ss= np.linspace(0, time_end, num_steps+1)
# # print(time_ss.shape)
# plt.figure(figsize=(10, 6))
# plt.plot(time_ss, midpoint_temperature_history, label='Midpoint Temperature')
# plt.axhline(y=T_L, color='r', linestyle='--', label='Liquidus Temperature')
# plt.axhline(y=T_S, color='g', linestyle='--', label='Solidus Temperature')
# plt.xlabel('Time(s)')
# plt.ylabel('Temperature (K)')
# plt.title('Temperature Distribution Over Time at x = 7.5mm') 
# plt.legend()
# plt.show()

# %% [markdown]
# 

# %% [markdown]
# ### Data into Array

# %%
temperature_history = np.array(temperature_history)

phi_history = np.array(phi_history)

t_hist = np.array(temperature_history[:,1:-1])
p_hist = np.array(phi_history[:,1:-1])

t_hist_init = t_hist[0,:]
t_hist_bc_l = t_hist[:,0]
t_hist_bc_r = t_hist[:,-1]





# %% [markdown]
# 

# %%
# Assuming you have temperature_history and phi_history as lists of arrays


# # Check the new shape after transposing
# print("Transposed Temperature History Shape:", temperature_history.shape)
# print("Transposed Phi History Shape:", phi_history.shape)

# # Create a meshgrid for space and time coordinates
# space_coord, time_coord = np.meshgrid(np.arange(temperature_history.shape[1]), np.arange(temperature_history.shape[0]))

# time_coord = time_coord * dt 
# # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # Plot the temperature history on the left subplot
# im1 = ax1.pcolormesh(space_coord, time_coord, temperature_history, cmap='viridis')
# ax1.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=16)
# ax1.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
# ax1.set_title('Temperature Variation Over Time',fontname='Times New Roman', fontsize=20)
# fig.colorbar(im1, ax=ax1, label='Temperature')

# # Plot the phase history on the right subplot
# im2 = ax2.pcolormesh(space_coord, time_coord, phi_history, cmap='viridis')
# ax2.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=18)
# ax2.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
# ax2.set_title('Phase Variation Over Time',fontname='Times New Roman', fontsize=20)
# fig.colorbar(im2, ax=ax2, label='Phase')
# plt.tight_layout()
# plt.show()

# #plot the main
# fig, ax = plt.subplots(figsize=(14, 6))
# im = ax.pcolormesh(space_coord, time_coord, Dim_ny, cmap='viridis')
# ax.set_xlabel('Space Coordinate')
# ax.set_ylabel('Time')
# ax.set_title('Niyama Variation Over Time')
# fig.colorbar(im, ax=ax, label='Main')
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ## ML training

# %% [markdown]
# ### GPU/CPU check

# %%
# check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %% [markdown]
# ### Data preparation

# %%



space = np.linspace(0, length, num_points) # Spatial points
time = np.linspace(0, time_end, num_steps+1) # Time points
print(space.shape,time.shape)

sp_i = np.linspace(0, length, num_points) # Spatial points
time_i = np.zeros(num_points) # Time points

sp_b_l = np.zeros(num_steps+1) # Spatial points
time_b_l = np.linspace(0, time_end, num_steps+1) # Time points

sp_b_r = np.ones(num_steps+1)*length # Spatial points
time_b_r = np.linspace(0, time_end, num_steps+1) # Time points




# %%

inputs = input_gen(space,time,'mgrid')
inputs_i = input_gen(sp_i,time_i,'scr')
inputs_b_l = input_gen(sp_b_l,time_b_l,'scr')
inputs_b_r = input_gen(sp_b_r,time_b_r,'scr')



# %%
print(inputs.shape,inputs_i.shape,inputs_b_l.shape,inputs_b_r.shape)

# %%


inputs = torch.tensor(inputs).float().to(device) # Convert the inputs to a tensor


inputs_init = torch.tensor(inputs_i).float().to(device) # Convert the inputs to a tensor
inputs_b_l = torch.tensor(inputs_b_l).float().to(device) # Convert the inputs to a tensor
inputs_b_r = torch.tensor(inputs_b_r).float().to(device) # Convert the inputs to a tensor

print(inputs_init.shape)
# label/temp data
temp_tr = torch.tensor(t_hist).float().to(device) # Convert the temperature history to a tensor
temp_inp = temp_tr.reshape(-1,1) # Reshape the temperature tensor to a column vector
temp_inp_init = torch.tensor(t_hist_init).float().to(device) # Convert the temperature history to a tensor
temp_inp_bc_l = torch.tensor(t_hist_bc_l).float().to(device) # Convert the temperature history to a tensor
temp_inp_bc_r = torch.tensor(t_hist_bc_r).float().to(device) # Convert the temperature history to a tensor
print(temp_inp.shape)



#Data Splitting

# train_inputs, val_test_inputs, train_temp_inp, val_test_temp_inp = train_test_split(inputs, temp_inp, test_size=0.2, random_state=42)
# val_inputs, test_inputs, val_temp_inp, test_temp_inp = train_test_split(val_test_inputs, val_test_temp_inp, test_size=0.8, random_state=42)

train_inputs, test_inputs, train_temp_inp, test_temp_inp = train_test_split(inputs, temp_inp, test_size=0.2, random_state=42)
train_inputs_init, test_inputs_init, train_temp_inp_init, test_temp_inp_init = train_test_split(inputs_init, temp_inp_init, test_size=0.2, random_state=42)
train_inputs_bc_l, test_inputs_bc_l, train_temp_inp_bc_l, test_temp_inp_bc_l = train_test_split(inputs_b_l, temp_inp_bc_l, test_size=0.2, random_state=42)
train_inputs_bc_r, test_inputs_bc_r, train_temp_inp_bc_r, test_temp_inp_bc_r = train_test_split(inputs_b_r, temp_inp_bc_r, test_size=0.2, random_state=42)




# %% [markdown]
# ### Create Data loader

# %%
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, temp_inp,transform=None, target_transform =None):
        self.inputs = inputs
        self.temp_inp = temp_inp

    def __getitem__(self, index):
        return self.inputs[index], self.temp_inp[index]
    
    def __len__(self):
        return len(self.inputs)

  
train_dataset = TensorDataset(train_inputs, train_temp_inp) # Create the training dataset
# val_dataset = TensorDataset(val_inputs, val_temp_inp) # Create the validation dataset
test_dataset = TensorDataset(test_inputs, test_temp_inp) # Create the test dataset

train_dataset_init = TensorDataset(train_inputs_init, train_temp_inp_init) # Create the training dataset
test_dataset_init = TensorDataset(test_inputs_init, test_temp_inp_init) # Create the test dataset
train_dataset_bc_l = TensorDataset(train_inputs_bc_l, train_temp_inp_bc_l) # Create the training dataset
test_dataset_bc_l = TensorDataset(test_inputs_bc_l, test_temp_inp_bc_l) # Create the test dataset
train_dataset_bc_r = TensorDataset(train_inputs_bc_r, train_temp_inp_bc_r) # Create the training dataset
test_dataset_bc_r = TensorDataset(test_inputs_bc_r, test_temp_inp_bc_r) # Create the test dataset


batch_size = 128

random_sampler_train = RandomSampler(train_dataset, replacement=True, num_samples=batch_size) # Create a random sampler for the training dataset
# random_sampler_val = RandomSampler(val_dataset, replacement=True, num_samples=batch_size) # Create a random sampler for the validation dataset
random_sampler_test = RandomSampler(test_dataset, replacement=True, num_samples=batch_size) # Create a random sampler for the test dataset

random_sampler_train_init = RandomSampler(train_dataset_init, replacement=True, num_samples=batch_size) # Create a random sampler for the training dataset
random_sampler_test_init = RandomSampler(test_dataset_init, replacement=True, num_samples=batch_size) # Create a random sampler for the test dataset
random_sampler_train_bc_l = RandomSampler(train_dataset_bc_l, replacement=True, num_samples=batch_size) # Create a random sampler for the training dataset
random_sampler_test_bc_l = RandomSampler(test_dataset_bc_l, replacement=True, num_samples=batch_size) # Create a random sampler for the test dataset
random_sampler_train_bc_r = RandomSampler(train_dataset_bc_r, replacement=True, num_samples=batch_size) # Create a random sampler for the training dataset
random_sampler_test_bc_r = RandomSampler(test_dataset_bc_r, replacement=True, num_samples=batch_size) # Create a random sampler for the test dataset





train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler_train) # Create the training dataloader
# val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=random_sampler_val) # Create the validation dataloader
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=random_sampler_test) # Create the test dataloader

train_loader_init = DataLoader(train_dataset_init, batch_size=batch_size, sampler=random_sampler_train_init) # Create the training dataloader
test_loader_init = DataLoader(test_dataset_init, batch_size=batch_size, sampler=random_sampler_test_init) # Create the test dataloader
train_loader_bc_l = DataLoader(train_dataset_bc_l, batch_size=batch_size, sampler=random_sampler_train_bc_l) # Create the training dataloader
test_loader_bc_l = DataLoader(test_dataset_bc_l, batch_size=batch_size, sampler=random_sampler_test_bc_l) # Create the test dataloader
train_loader_bc_r = DataLoader(train_dataset_bc_r, batch_size=batch_size, sampler=random_sampler_train_bc_r) # Create the training dataloader
test_loader_bc_r = DataLoader(test_dataset_bc_r, batch_size=batch_size, sampler=random_sampler_test_bc_r) # Create the test dataloader



# %% [markdown]
# ### NN Architecture Definition

# %%

# Define the neural network architecture
class Mushydata(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # This is the constructor
        super(Mushydata, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x, t):                               # This is the forward pass
        input_features = torch.cat([x, t], dim=1)          # Concatenate the input features
        m = self.base(input_features)                                 # Pass through the third layer
        return m                    # Return the output of the network


# features = torch.rand(1, 2)
# model = HeatPINN(2, 20, 1)
# output = model(features[:, 0:1], features[:, 1:2])
# print(output)


# Loss function for data 


# %% [markdown]
# ### Hyperparamters Init

# %%
# Hyperparameters
hidden_size = 40
learning_rate = 0.004
epochs = 30000
# alpha = 0.01  # Adjust this value based on your problem
# boundary_value = 313.0
# initial_value = init_temp
# Initialize the model
model = Mushydata(input_size=2, hidden_size=hidden_size,output_size=1).to(device)
lambd = 0.1

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# %% [markdown]
# ### Loss List Init

# %%
train_losses = []
val_losses = []
test_losses = []

print(f"Datatype of train_loader is {type(train_loader)}")

# %% [markdown]
# ### Loss functions

# %%
# def loss_fn_data(u_pred, u_true):
#     return nn.MSELoss()(u_pred, u_true)

# def l1_regularization(model, lambd):
#     l1_reg = sum(param.abs().sum() for param in model.parameters())
#     return l1_reg * lambd

# def pde_loss(u_pred,x,t):
#     # u_pred.requires_grad = True
#     x.requires_grad = True
#     t.requires_grad = True
    
#     u_pred = model(x,t).requires_grad_()
#     u_t = torch.autograd.grad(u_pred, t, 
#                                 torch.ones_like(u_pred).to(device),
#                                 create_graph=True,
#                                 allow_unused=True,
#                                 )[0] # Calculate the first time derivative
#     if u_t is None:
#         raise RuntimeError("u_t is None")

#     u_x = torch.autograd.grad(u_pred, 
#                                 x, 
#                                 torch.ones_like(u_pred).to(device), 
#                                 create_graph=True,
#                                 allow_unused =True)[0] # Calculate the first space derivative
            
#     u_xx = torch.autograd.grad(u_x, 
#                                 x, 
#                                 torch.ones_like(u_x).to(device), 
#                                 create_graph=True,
#                                 allow_unused=True)[0] 
    
#     T_S_tensor = torch.tensor(T_S, device=device)
#     T_L_tensor = torch.tensor(T_L, device=device)
    
#     k_m = torch.where((u_pred >= T_S_tensor) * (u_pred <= T_L_tensor),\
#                        kramp(u_pred, k_l,k_s,T_L,T_S),torch.tensor(0.0,device=device))
#     cp_m = torch.where(u_pred >= T_S_tensor * u_pred <= T_L_tensor, cp_ramp((u_pred), cp_l,cp_s,T_L,T_S))
#     rho_m = torch.where(u_pred >= T_S_tensor * u_pred <= T_L_tensor, rho_ramp((u_pred), rho_l,rho_s,T_L,T_S))
#     m_eff = (k_m / (rho_m * (cp_m + (L_fusion / (T_L - T_S)))))

#     alpha_T = torch.where(u_pred >= T_L_tensor, alpha_l, torch.where(u_pred<=T_S_tensor,alpha_s ,m_eff))
#     alpha_T = 1
#     residual = u_t - alpha_T * u_xx

#     return nn.MSELoss()(residual,torch.zeros_like(residual))

# def boundary_loss(u_pred,x,t,t_surr):
    
#     u_x = torch.autograd.grad(u_pred,x, 
#                                 torch.ones_like(u_pred).to(device), 
#                                 create_graph=True,
#                                 allow_unused =True)[0] # Calculate the first space derivative
#     t_surr_t = torch.tensor(t_surr, device=device)
#     res_l = u_x -(htc* (u_pred-t_surr_t))
   

#     return nn.MSELoss()(res_l,torch.zeros_like(res_l))

# def ic_loss(u_pred):
#     temp_init_tsr = torch.tensor(temp_init[1:-1],device=device)
#     ic = u_pred -temp_init_tsr
#     return nn.MSELoss()(ic,torch.zeros_like(ic))

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true) / u_true)

# %% [markdown]
# ### Training, Validation and Testing Module

# %%
def training_loop(epochs, model, loss_fn_data, optimizer, train_dataloader,):
    train_losses = []  # Initialize the list to store the training losses
    val_losses = []    # Initialize the list to store the validation losses

    for epoch in range(epochs):
        model.train()                                                                           # Set the model to training mode
        train_loss = 0                                                                              # Initialize the training loss
        train_accuracy = 0
        for (batch,batch_init,batch_left,batch_right) in \
             zip (train_dataloader,train_loader_init,train_loader_bc_l,train_inputs_bc_r):                                                          # Loop through the training dataloader
            inputs, temp_inp= batch                                                             # Get the inputs and the true values
            inputs_init, temp_inp_init= batch_init                                                             # Get the inputs and the true values 
            inputs_left, temp_inp_left= batch_left                                                             # Get the inputs and the true values
            inputs_right, temp_inp_right= batch_right                                                             # Get the inputs and the true values

            inputs, temp_inp= inputs.to(device), temp_inp.to(device)                             # Move the inputs and true values to the GPU
            inputs_init, temp_inp_init= inputs_init.to(device), temp_inp_init.to(device)                             # Move the inputs and true values to the GPU
            inputs_left, temp_inp_left= inputs_left.to(device), temp_inp_left.to(device)                             # Move the inputs and true values to the GPU
            inputs_right, temp_inp_right= inputs_right.to(device), temp_inp_right.to(device)                             # Move the inputs and true values to the GPU

            optimizer.zero_grad()                                                                    # Zero the gradients
            
            # Forward pass
            u_pred = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1)).to(device)                       # Get the predictions
            u_initl = model(inputs_init[:,0].unsqueeze(1), inputs_init[:,1].unsqueeze(1)).to(device)                       # Get the predictions
            
            u_left = model(inputs_b_l[:,0].unsqueeze(1), inputs_b_l[:,1].unsqueeze(1)).to(device)               # Left boundary of the temperature
            u_right = model(inputs_b_r[:,0].unsqueeze(1), inputs_b_r[:,1].unsqueeze(1)).to(device)             # Right boundary of the temperature

            # Loss calculation
            data_loss = loss_fn_data(u_pred, temp_inp)                                              # Calculate the data loss
            
            pd_loss = pde_loss(model,inputs[:,0].unsqueeze(1),inputs[:,1].unsqueeze(1))             # Calculate the PDE loss
            # pd_loss = 0
            
            initc_loss = ic_loss(u_initl) 
            # initc_loss =0                                                      # Calculate initial condition loss
            
            bc_loss_left = boundary_loss(model,inputs_b_l[:,0].unsqueeze(1),inputs_b_l[:,1].unsqueeze(1),t_surr) # Calculate the left boundary condition loss
            bc_loss_right = boundary_loss(model,inputs_b_r[:,0].unsqueeze(1),inputs_b_r[:,1].unsqueeze(1),t_surr) # Calculate the right boundary condition loss
            bc_loss = bc_loss_left + bc_loss_right
            # l1_regularization_loss = l1_regularization(model, lambda_l1)                      # Calculate the L1 regularization loss
            # loss = data_loss  + pd_loss + initc_loss + bc_loss                                              # Calculate the total loss
            loss = data_loss + pd_loss + initc_loss + bc_loss
            train_accuracy += accuracy(u_pred, temp_inp)                                                              # Calculate the total loss
            # Backpropagation
            loss.backward(retain_graph=True)                                                        # Backpropagate the gradients
            
            optimizer.step()                                                                           # Update the weights
            
            train_loss += loss.item()                                                           # Add the loss to the training set loss                 

        

        # model.eval()
        # test_loss = 0
        # test_accuracy = 0
        # with torch.no_grad():   
        #     for batch in test_dataloader:
        #         inputs, temp_inp= batch
        #         inputs, temp_inp= inputs.to(device), temp_inp.to(device)
        #         u_pred = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1))
        #         data_loss = loss_fn_data(u_pred, temp_inp)
        #         # l1_regularization_loss = l1_regularization(model, lambd)
        #         # loss = data_loss  + l1_regularization_loss
        #         loss = data_loss
        #         test_accuracy = accuracy(u_pred, temp_inp)
        #         test_loss += loss.item()
        #     test_losses.append(test_loss)

        train_losses.append(train_loss)                                                   # Append the training loss to the list of training losses
        
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Training-Loss {train_loss:.4e}")
        
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training-Loss {train_loss:.4e}, Data-loss {data_loss:.4e}\
                  , pde-loss {pd_loss:.4e}, initc-loss {initc_loss:.4e}\
                    bc_loss {bc_loss:.4e}") 

    return train_losses, val_losses                                                             # Return the training and validation losses


# %%
def test_loop(epochs, model, loss_fn_data, optimizer, train_dataloader, test_dataloader):
      
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():   
        for batch in test_dataloader:
            inputs, temp_inp= batch
            inputs, temp_inp= inputs.to(device), temp_inp.to(device)
            u_pred = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1))
            data_loss = loss_fn_data(u_pred, temp_inp)
            # l1_regularization_loss = l1_regularization(model, lambd)
            # loss = data_loss  + l1_regularization_loss
            loss = data_loss
            test_accuracy = accuracy(u_pred, temp_inp)
            test_loss += loss.item()
        test_losses.append(test_loss)
    if epochs % 10 == 0:
        print(f"Epoch {epochs}, Test-Loss {test_loss:.4e}, Test-Accuracy {test_accuracy:.4e}")      
    return test_losses

# %% [markdown]
# ### Training Button 

# %%

train_losses, val_losses = training_loop(epochs, model, loss_fn_data, optimizer,train_loader)  # Train the model
 
test_losses = test_loop(epochs, model, loss_fn_data, optimizer, train_loader, test_loader)  # Test the model


   


    
    

# %%
temp_nn = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1)).cpu().detach().numpy() # Get the predictions from the model

temp_nn = temp_nn.reshape(num_steps+1, num_points) # Reshape the predictions to a 2D array
time_ss= np.linspace(0, time_end, num_steps+1)
plt.figure
plt.plot(time_ss, temp_nn[:,num_points//2], label='Predicted Temperature')
plt.plot(time_ss, temperature_history[:,num_points//2], label='Actual Temperature')
plt.xlabel('Time(s)')
plt.ylabel('Temperature (K)')
plt.title('Predicted vs Actual Temperature at x = 7.5mm')
plt.legend()
plt.show()


# %%
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


