
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Material properties
rho = 2300.0                     # Density of AL380 (kg/m^3)
rho_l = 2460.0                   # Density of AL380 (kg/m^3)
rho_l_t = torch.tensor(rho_l,device=device)
rho_s = 2710.0                    # Density of AL380 (kg/m^3)
rho_s_t = torch.tensor(rho_s,device=device)
rho_m = (rho_l + rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

k = 104.0                       # W/m-K
k_l = k                       # W/m-K
k_l_t = torch.tensor(k_l,device=device)
k_s = 96.2                    # W/m-K
k_s_t = torch.tensor(k_s,device=device)
k_m =  (k_l+k_s)/2                     # W/m-K
k_mo = 41.5
k_mo_t = torch.tensor(k_mo,device=device)


cp = 1245.3                      # Specific heat of aluminum (J/kg-K)
cp_l = cp                      # Specific heat of aluminum (J/kg-K)
cp_l_t = torch.tensor(cp_l,device=device)
cp_s = 963.0                 # Specific heat of aluminum (J/kg-K)
cp_s_t = torch.tensor(cp_s,device=device)
cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity
alpha_l = k_l / (rho_l * cp_l) 
alpha_l_t = torch.tensor(alpha_l,device=device)
alpha_s = k_s / (rho_s*cp_s)
alpha_s_t = torch.tensor(alpha_s,device=device)
alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`


#L_fusion = 3.9e3                 # J/kg
L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum
         # Thermal diffusivity
L_fusion_t = torch.tensor(L_fusion,device=device)
t_surr = 500.0 
t_surr_t = torch.tensor(t_surr,device=device)
temp_init = 919.0
temp_init_t = torch.tensor(temp_init,device=device)
T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380

T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)




def kramp(temp,v1,v2,T_L,T_s):                                      # Function to calculate thermal conductivity in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        k_m = torch.where(
        temp > T_L,
        torch.tensor(v1,device=temp.device),
        torch.where(
            temp < T_S,
            torch.tensor(v2,device=temp.device),
            torch.tensor(v2,device=temp.device) + slope*(temp-T_S),
            )
            ) 
        k_m_t = torch.tensor(k_m,device=temp.device)  
        return k_m_t

def cp_ramp(temp,v1,v2,T_L,T_S):                                    # Function to calculate specific heat capacity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    cp_m = torch.where(
        temp > T_L,
        torch.tensor(v1,device=temp.device),
        torch.where(
            temp < T_S,
            torch.tensor(v2,device=temp.device),
            torch.tensor(v2,device=temp.device) + slope*(temp-T_S),
        )
    )
    cp_m_t = torch.tensor(cp_m,device=temp.device)
    return cp_m_t

def rho_ramp(temp,v1,v2,T_L,T_S):                                       # Function to calculate density in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    rho_m = torch.where(
        temp > T_L,
        torch.tensor(v1,device=temp.device),
        torch.where(
            temp < T_S,
            torch.tensor(v2,device=temp.device),
            torch.tensor(v2,device=temp.device) + slope*(temp-T_S),
        )
    )
    rho_m_t = torch.tensor(rho_m,device=temp.device)
    return rho_m_t





def loss_fn_data(u_pred, u_true):
    return nn.MSELoss()(u_pred, u_true)

def l1_regularization(model, lambd):
    l1_reg = sum(param.abs().sum() for param in model.parameters())
    return l1_reg * lambd

def pde_loss(model,x,t):
    # u_pred.requires_grad = True
    x.requires_grad = True
    t.requires_grad = True
    
    u_pred = model(x,t).requires_grad_()
    u_t = torch.autograd.grad(u_pred, t, 
                                torch.ones_like(u_pred).to(device),
                                create_graph=True,
                                allow_unused=True,
                                )[0] # Calculate the first time derivative
    if u_t is None:
        raise RuntimeError("u_t is None")

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
    
    k_m = torch.where((u_pred >= T_S_tensor) * (u_pred <= T_L_tensor),\
                       kramp(u_pred, k_l_t,k_s_t,T_L_tensor,T_S_tensor),\
                           torch.tensor(0.0,device=device))
    cp_m = torch.where((u_pred >= T_S_tensor) * (u_pred <= T_L_tensor), cp_ramp((u_pred),\
                          cp_l_t,cp_s_t,T_L_tensor,T_S_tensor),\
                       torch.tensor(0.0,device=device))
    rho_m = torch.where((u_pred >= T_S_tensor) * (u_pred <= T_L_tensor), rho_ramp((u_pred), \
                                 rho_l_t,rho_s_t,T_L_tensor,T_S_tensor),\
                        torch.tensor(0.0,device=device))
    m_eff = (k_m / (rho_m * (cp_m + (L_fusion_t / (T_L_tensor - T_S_tensor)))))

    alpha_T = torch.where(u_pred >= T_L_tensor, alpha_l, \
                             torch.where(u_pred<=T_S_tensor,alpha_s ,m_eff))
    # alpha_T = 1
    residual = u_t - alpha_T * u_xx

    return nn.MSELoss()(residual,torch.zeros_like(residual))

def boundary_loss(model,x,t,t_surr_t):
    
    x.requires_grad = True
    t.requires_grad = True
    
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
   

    return nn.MSELoss()(res_l,torch.zeros_like(res_l))

def ic_loss(u_pred):
    temp_init_tsr = torch.tensor(temp_init,device=device)
    ic = u_pred -temp_init_tsr
    return nn.MSELoss()(ic,torch.zeros_like(ic))

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true) / u_true)