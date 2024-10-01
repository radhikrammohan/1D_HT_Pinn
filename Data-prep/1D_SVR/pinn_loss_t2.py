
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Material properties

rho = 2710.0                    # Density of AL380 (kg/m^3)
rhot = torch.tensor(rho,dtype=torch.float32,device=device)

# rho_m = (rho_l + rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density


k = 96.2                    # W/m-K
k_t = torch.tensor(k,dtype=torch.float32,device=device)
# k_m =  (k_l+k_s)/2                     # W/m-K
k_mo = 41.5



cp = 963.0                 # Specific heat of aluminum (J/kg-K)
cp_t = torch.tensor(cp,dtype=torch.float32,device=device)
# cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity

alpha = k / (rho*cp)
alpha_t = torch.tensor(alpha,dtype=torch.float32,device=device)

# alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`


#L_fusion = 3.9e3                 # J/kg
L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum

L_fusion_t = torch.tensor(L_fusion,dtype=torch.float32,device=device)
         # Thermal diffusivity

# t_surr = 500.0 
# temp_init = 919.0
T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380
T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)
T_St = torch.tensor(T_S,dtype=torch.float32,device=device)
T_Lt = torch.tensor(T_L,dtype=torch.float32,device=device)







def loss_fn_data(u_pred, u_true):
    return nn.MSELoss()(u_pred, u_true)

def l1_regularization(model, lambd):
    l1_reg = sum(param.abs().sum() for param in model.parameters())
    return l1_reg * lambd

def pde_loss(model,x,t):
    # u_pred.requires_grad = True
    x.requires_grad = True
    t.requires_grad = True
    
    u_pred = model(x,t)
    u_pred = u_pred.unsqueeze(1).to(device)
    u_t = torch.autograd.grad(outputs=u_pred, inputs=t, 
                                grad_outputs=torch.ones_like(u_pred).to(device),
                                create_graph=True,                                
                                )[0] # Calculate the first time derivative
    if u_t is None:
        raise RuntimeError("u_t is None")

    u_x = torch.autograd.grad(outputs=u_pred, 
                                inputs=x, 
                               grad_outputs=torch.ones_like(u_pred).to(device), 
                                create_graph=True,
                                )[0] # Calculate the first space derivative
            
    u_xx = torch.autograd.grad(outputs=u_x, 
                                inputs=x, 
                                grad_outputs=torch.ones_like(u_x).to(device), 
                                create_graph=True,
                                )[0]
    
    
    
    
    residual = u_t - (alpha_t*u_xx)  

    # print(res_sq.dtype) 
    resid_mean = torch.mean(torch.square(residual)) 
    # print(resid_mean.dtype)
    # print(resid_mean)
    return resid_mean

def boundary_loss(model,x,t,t_surr):
    
    x.requires_grad = True
    t.requires_grad = True
    t_surr_t = torch.tensor(t_surr, device=device)
    u_pred = model(x,t).requires_grad_(True)
    u_x = torch.autograd.grad(u_pred,x, 
                                torch.ones_like(u_pred).to(device), 
                                create_graph=True,
                                allow_unused =True)[0] # Calculate the first space derivative
   
    htc =10.0
    if u_x is None:
        raise RuntimeError("u_x is None")
    if u_pred is None:
        raise RuntimeError("u_pred is None")
    if t_surr_t is None:
        raise RuntimeError("t_surr_t is None")
    res_l = u_x -(htc*(u_pred-t_surr_t))
    # print(res_l)

    return nn.MSELoss()(res_l,torch.zeros_like(res_l))

def ic_loss(u_pred,temp_init):
    temp_init_tsr = torch.tensor(temp_init,device=device)
   
    ic_mean = torch.mean(torch.square(u_pred -temp_init_tsr))
   
    return ic_mean

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true) / u_true)