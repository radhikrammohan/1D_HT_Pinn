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
from torch.utils.data import DataLoader, TensorDataset


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

# %%
# # Plot temperature history for debugging
# plt.figure(figsize=(10, 6))
# for i in range(0, num_steps, num_steps // 50):
#     plt.plot(np.linspace(0, length, num_points+2), temperature_history[i], label=f't={i * dt:.2f} s')

# plt.xlabel('Position (m)')
# plt.ylabel('Temperature (K)')
# plt.title('Temperature Distribution Over Time')
# # plt.legend()
# plt.show()

# %% [markdown]
# ### Data into Array

# %%
temperature_history = np.array(temperature_history)

phi_history = np.array(phi_history)

t_hist = np.array(temperature_history[:,1:-1])
p_hist = np.array(phi_history[:,1:-1])
print(t_hist.shape)
print(p_hist.shape)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

space = np.linspace(0, length, num_points)
time = np.linspace(0, time_end, num_steps+1)

scaler_space = StandardScaler()
scaler_time = StandardScaler()

space_tr = scaler_space.fit_transform(space.reshape(-1,1))
time_tr = scaler_time.fit_transform(time.reshape(-1,1))

print(space_tr.shape)
print(time_tr.shape)


# create mesh grid of space and time

space_tr, time_tr = np.meshgrid(space_tr, time_tr)
space_tr = space_tr.flatten().reshape(-1,1)
time_tr = time_tr.flatten().reshape(-1,1)
inputs = np.hstack([space_tr,time_tr]) # Concatenate the spatial and temporal inputs
inputs = torch.tensor(inputs).float().to(device) # Convert the inputs to a tensor
print(inputs.shape)

# label/temp data
temp_tr = torch.tensor(t_hist).float().to(device) # Convert the temperature history to a tensor
temp_inp = temp_tr.reshape(-1,1) # Reshape the temperature tensor to a column vector
print(temp_inp.shape)



#Data Splitting

train_inputs, val_test_inputs, train_temp_inp, val_test_temp_inp = train_test_split(inputs, temp_inp, test_size=0.2, random_state=42)
val_inputs, test_inputs, val_temp_inp, test_temp_inp = train_test_split(val_test_inputs, val_test_temp_inp, test_size=0.8, random_state=42)





# %% [markdown]
# ### Create Data loader

# %%
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]

        return X, y




train_dataset = TensorDataset(train_inputs, train_temp_inp) # Create the training dataset
val_dataset = TensorDataset(val_inputs, val_temp_inp) # Create the validation dataset
test_dataset = TensorDataset(test_inputs, test_temp_inp) # Create the test dataset

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create the training dataloader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) # Create the validation dataloader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # Create the test dataloader



# %% [markdown]
# ### NN Architecture Definition

# %%

# Define the neural network architecture
class Mushydata(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): # This is the constructor
        super(Mushydata, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
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
hidden_size = 200
learning_rate = 0.005
epochs = 3
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
def loss_fn_data(u_pred, u_true):
    return nn.MSELoss()(u_pred, u_true)

def l1_regularization(model, lambd):
    l1_reg = sum(param.abs().sum() for param in model.parameters())
    return l1_reg * lambd

def pde_loss(u_pred,x,t):
    # u_pred.requires_grad = True
    x.requires_grad = True
    t.requires_grad = True
    
    u_pred = u_pred.requires_grad_(True)
    u_t = torch.autograd.grad(u_pred, t, 
                                torch.ones_like(u_pred).to(device), 
                                create_graph=True, 
                                allow_unused=True
                                )[0] # Calculate the first time derivative
    # if u_t is None:
    #     raise RuntimeError("u_t is None")

    u_x = torch.autograd.grad(u_pred, 
                                x, 
                                torch.ones_like(u_pred).to(device), 
                                create_graph=True,
                                allow_unused =True)[0] # Calculate the first space derivative
            
    u_xx = torch.autograd.grad(u_x, 
                                x, 
                                torch.ones_like(u_x).to(device), 
                                create_graph=True,
                                allow_unused=True)[0] 
    
    T_S_tensor = torch.tensor(T_S, device=device)
    T_L_tensor = torch.tensor(T_L, device=device)
    k_m = torch.where(u_pred >= T_S_tensor & u_pred <= T_L_tensor, k_ramp((u_pred), k_l,k_s,T_L,T_S))
    cp_m = torch.where(u_pred >= T_S_tensor & u_pred <= T_L_tensor, cp_ramp((u_pred), cp_l,cp_s,T_L,T_S))
    rho_m = torch.where(u_pred >= T_S_tensor & u_pred <= T_L_tensor, rho_ramp((u_pred), rho_l,rho_s,T_L,T_S))
    m_eff = (k_m / (rho_m * (cp_m + (L_fusion / (T_L - T_S)))))

    alpha_T = torch.where(u_pred >= T_L_tensor, alpha_l, torch.where(u_pred<=T_S_tensor,alpha_s ,m_eff))

    residual = u_t - alpha_T * u_xx

    return nn.MSELoss()(residual,torch.zeros_like(residual))

# def boundary_loss(u_pred):
    
#     u_x = torch.autograd.grad(u_pred,x, 
#                                 torch.ones_like(u_pred).to(device), 
#                                 create_graph=True,
#                                 allow_unused =True)[0] # Calculate the first space derivative
    
#     res_l = htc* (model()
   

#     return nn.MSELoss()(bc_l,torch.zeros_like(bc_l)) + nn.MSELoss()(bc_r,torch.zeros_like(bc_r))

# def ic_loss(u_pred):
#     ic_r = u_pred -temp_init
#     return nn.MSELoss()(ic_r,torch.zeros_like(ic_r))

# %% [markdown]
# ### Training, Validation and Testing Module

# %%
def training_loop(epochs, model, loss_fn_data, optimizer, train_dataloader, val_dataloader):
    train_losses = []  # Initialize the list to store the training losses
    val_losses = []    # Initialize the list to store the validation losses

    for epoch in range(epochs):
        model.train()                                                           # Set the model to training mode
        train_loss = 0                                                       # Initialize the training loss

        for batch in train_dataloader:                                             # Loop through the training dataloader
            inputs, temp_inp= batch                                               # Get the inputs and the true values
            inputs, temp_inp= inputs.to(device), temp_inp.to(device)             # Move the inputs and true values to the GPU
            optimizer.zero_grad()                                                       # Zero the gradients
            
            # Forward pass
            u_pred = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1)).to(device) # Get the predictions
            
            # Loss calculation
            data_loss = loss_fn_data(u_pred, temp_inp)                                  # Calculate the data loss
            pd_loss = pde_loss(u_pred,inputs[:,0].unsqueeze(1),inputs[:,1].unsqueeze(1)) # Calculate the PDE loss
            
            # l1_regularization_loss = l1_regularization(model, lambda_l1)                  # Calculate the L1 regularization loss
            loss = data_loss  +pde_loss                                                                       # Calculate the total loss
            
            # Backpropagation
            loss.backward(retain_graph=True)                                                        # Backpropagate the gradients
            
            optimizer.step()                                                                            # Update the weights
            
            train_loss += loss.item()     
                                        # Add the loss to the training set loss
        train_losses.append(train_loss)                                                   # Append the training loss to the list of training losses
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training-Loss {train_loss:.4e}")
        
        model.eval()
        val_loss = 0                                                                                      # Initialize the validation loss
        
        with torch.no_grad():
            for batch in val_dataloader:                                                            # Loop through the validation dataloader
                inputs, temp_inp= batch                                                        # Get the inputs and the true values
                inputs, temp_inp= inputs.to(device), temp_inp.to(device)                                     # Move the inputs and true values to the GPU
                u_pred = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1))                        # Get the predictions
                data_loss = loss_fn_data(u_pred, temp_inp)                                          # Calculate the data loss
                loss = data_loss                                                                          # Calculate the total loss
                val_loss += loss.item()                                                                     # Add the loss to the validation set loss
            val_losses.append(val_loss)                                                                 # Append the validation loss to the list of validation losses
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Validation-Loss {val_loss:.4e}")                                                           
    return train_losses, val_losses                                                                    # Return the training and validation losses


# %%
def test_loop(epochs, model, loss_fn_data, optimizer, train_dataloader, test_dataloader):
      
    model.eval()
    test_loss = 0
    with torch.no_grad():   
        for batch in test_dataloader:
            inputs, temp_inp= batch
            inputs, temp_inp= inputs.to(device), temp_inp.to(device)
            u_pred = model(inputs[:,0].unsqueeze(1), inputs[:,1].unsqueeze(1))
            data_loss = loss_fn_data(u_pred, temp_inp)
            loss = data_loss 
            test_loss += loss.item()
        test_losses.append(test_loss)
    if epochs % 10 == 0:
        print(f"Epoch {epochs}, Test-Loss {test_loss:.4e}")    
    return test_losses

# %% [markdown]
# ### Training Button 

# %%

train_losses, val_losses = training_loop(epochs, model, loss_fn_data, optimizer,train_loader, val_loader)  # Train the model
 
test_losses = test_loop(epochs, model, loss_fn_data, optimizer, train_loader, test_loader)  # Test the model


   


    
    


