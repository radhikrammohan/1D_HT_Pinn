
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn import svm
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

print('Using device:', device)

# Material properties
rho = 2300.0                     # Density of AL380 (kg/m^3)
rho_l = 2460.0                   # Density of AL380 (kg/m^3)
rho_l_t = torch.tensor(rho_l,dtype=torch.float32,device=device)
rho_s = 2710.0                    # Density of AL380 (kg/m^3)
rho_s_t = torch.tensor(rho_s,dtype=torch.float32,device=device)

# rho_m = (rho_l + rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

k = 104.0                       # W/m-K
k_l = k                       # W/m-K
k_l_t = torch.tensor(k_l,dtype=torch.float32,device=device)
k_s = 96.2                    # W/m-K
k_s_t = torch.tensor(k_s,dtype=torch.float32,device=device)
# k_m =  (k_l+k_s)/2                     # W/m-K
k_mo = 41.5


cp = 1245.3                      # Specific heat of aluminum (J/kg-K)
cp_l = cp                      # Specific heat of aluminum (J/kg-K)
cp_l_t = torch.tensor(cp_l,dtype=torch.float32,device=device)
cp_s = 963.0                 # Specific heat of aluminum (J/kg-K)
cp_s_t = torch.tensor(cp_s,dtype=torch.float32,device=device)
# cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity
alpha_l = k_l / (rho_l * cp_l) 
alpha_l_t = torch.tensor(alpha_l,dtype=torch.float32,device=device)
# alpha_s = k_s / (rho_s*cp_s)
# alpha_s_t = torch.tensor(alpha_s,dtype=torch.float32,device=device)

# # alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`


# #L_fusion = 3.9e3                 # J/kg
# L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum

# L_fusion_t = torch.tensor(L_fusion,dtype=torch.float32,device=device)
#          # Thermal diffusivity

# t_surr = 500.0 
# temp_init = 919.0
T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380
T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)
T_St = torch.tensor(T_S,dtype=torch.float32,device=device)
T_Lt = torch.tensor(T_L,dtype=torch.float32,device=device)



def kramp(temp,v1,v2,T_L,T_S):              # Function to calculate thermal conductivity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    
    k_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
        
    return k_m

def cp_ramp(temp,v1,v2,T_L,T_S):        # Function to calculate specific heat capacity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    cp_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
    return cp_m

def rho_ramp(temp,v1,v2,T_L,T_S):         # Function to calculate density in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    rho_m = torch.where(temp > T_L, v1, torch.where(temp < T_S, v2, v2 + slope*(temp-T_S)))
    
    return rho_m





def loss_fn_data(u_pred, u_true):
    return nn.MSELoss()(u_pred, u_true)

def l1_regularization(model, lambd):
    l1_reg = sum(param.abs().sum() for param in model.parameters())
    return l1_reg * lambd

def pde_loss(model,x,t,T_S,T_L):
    # u_pred.requires_grad = True
    x.requires_grad = True
    t.requires_grad = True
    
    u_pred = model(x,t).to(device)
    # u_pred  = model

    u_t = torch.autograd.grad(u_pred, t, 
                                torch.ones_like(u_pred),
                                create_graph=True,
                                allow_unused=True,
                                )[0] # Calculate the first time derivative
    if u_t is None:
        raise RuntimeError("u_t is None") # Check if u_t is None

    u_x = torch.autograd.grad(u_pred, 
                                x, 
                                torch.ones_like(u_pred), 
                                create_graph=True,
                                allow_unused =True)[0] # Calculate the first space derivative

    if u_x is None:
        raise RuntimeError("u_x is None") # Check if u_x is None
           
    u_xx = torch.autograd.grad(u_x, 
                                x, 
                                torch.ones_like(u_x), 
                                create_graph=True,
                                allow_unused=True,
                                materialize_grads=True)[0]
    
    if u_xx is None:
        raise RuntimeError("u_xx is None") # Check if u_xx is None

    # T_S_tensor = T_S.clone().detach().to(device)
    # T_L_tensor = T_L.clone().detach().to(device)
    
    residual = u_t - (u_xx) # Calculate the residual of the PDE
   
    resid_mean = torch.mean(torch.square(residual))
    # resid_mean = nn.MSELoss()(residual,torch.zeros_like(residual).to(device))
    # print(resid_mean.dtype)
    
    return resid_mean

def boundary_loss(model,x,t,t_surr,t_init):
    
    # x.requires_grad = True
    # t.requires_grad = True
    # t_surr_t = torch.tensor(t_surr, device=device)
    # u_pred = model(x,t).requires_grad_(True)
    # u_x = torch.autograd.grad(u_pred,x, 
    #                             torch.ones_like(u_pred).to(device), 
    #                             create_graph=True,
    #                             allow_unused =True)[0] # Calculate the first space derivative
   
    # htc =10.0
    # if u_x is None:
    #     raise RuntimeError("u_x is None")
    # if u_pred is None:
    #     raise RuntimeError("u_pred is None")
    # if t_surr_t is None:
    #     raise RuntimeError("t_surr_t is None")
    # res_l = u_x -(htc*(u_pred-t_surr_t))
    
    # t_surr_t = t_surr.clone().detach().to(device)
    
    # def bc_func(x,t,t_surr,t_init):
    #     if (t == 0).any():
    #         return t_init
    #     else:
    #         return t_surr
        
    u_pred = model(x,t)
    bc = torch.where(t == 0, t_init, t_surr)
    
    bc_mean =  torch.mean(torch.square(u_pred-bc))
    # bc_mean = nn.MSELoss()(u_pred,bc)
   
    return bc_mean

def ic_loss(model,x,t,temp_init):
    
    u_pred = model(x,t)
    
    # def ic_func(x,t,temp_init):
    #     return temp_init
    
    # u_ic = ic_func(x,t,temp_init)
    
    # # u_del = u_pred - temp_init
    temp_i = torch.full_like(u_pred,temp_init)
   
    # ic_mean = nn.MSELoss()(u_pred,temp_i)    
    ic_mean = torch.mean(torch.square(u_pred-temp_i))
    return ic_mean

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true))