
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
alpha_s = k_s / (rho_s*cp_s)
alpha_s_t = torch.tensor(alpha_s,dtype=torch.float32,device=device)

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
                                allow_unused=True,
                                materialize_grads=True)[0][:, 0:1]
    
    T_S_tensor = torch.tensor(T_S, dtype=torch.float32, device=device)
    T_L_tensor = torch.tensor(T_L, dtype=torch.float32, device=device)
    
    # print(u_pred.dtype)
    # print(u_t.dtype)
    # print(u_x.dtype)    
    # print(u_xx.dtype)
    
    residual = torch.zeros_like(u_pred)
    alpha_T = torch.zeros_like(u_pred)
    k_m = torch.zeros_like(u_pred)
    cp_m = torch.zeros_like(u_pred)
    rho_m = torch.zeros_like(u_pred)

    

    alpha_s = (alpha_s_t * T_S_tensor) / T_St
    
    alpha_l = (alpha_l_t * T_L_tensor) / T_Lt

    

    mask_solid = u_pred <= T_S_tensor
    mask_liquid = u_pred >= T_L_tensor
    mask_mushy = ~(mask_solid | mask_liquid)

    alpha_T_solid = alpha_s
    residual[mask_solid] = u_t[mask_solid] - (alpha_T_solid * u_xx[mask_solid])

    alpha_T_liquid = alpha_l
    residual[mask_liquid] = u_t[mask_liquid] - (alpha_T_liquid * u_xx[mask_liquid])

    k_m_mushy = kramp(u_pred[mask_mushy], k_l_t, k_s_t, T_L_tensor, T_S_tensor)
    cp_m_mushy = cp_ramp(u_pred[mask_mushy], cp_l_t, cp_s_t, T_L_tensor, T_S_tensor)
    rho_m_mushy = rho_ramp(u_pred[mask_mushy], rho_l_t, rho_s_t, T_L_tensor, T_S_tensor)
    
    u1 = L_fusion_t / (T_L_tensor - T_S_tensor)
    alpha_T_mushy = (k_m_mushy / (rho_m_mushy * (cp_m_mushy + (u1))))

    residual[mask_mushy] = u_t[mask_mushy] - (alpha_T_mushy * u_xx[mask_mushy])

    
    # for i in range(len(u_pred)):
    #     if u_pred[i] < T_S_tensor:
    #        k_m[i] = k_s_t
    #        cp_m[i] = cp_s_t
    #        rho_m[i] = rho_s_t
    #        alpha_T[i] = alpha_s_t
    #        residual[i] = u_t[i] - (alpha_T[i] * u_xx[i])

    #     elif u_pred[i] >= T_L_tensor:
    #         k_m[i] = k_l_t
    #         cp_m[i] = cp_l_t
    #         rho_m[i] = rho_l_t
    #         alpha_T[i] = alpha_l_t
    #         residual[i] = u_t[i] - (alpha_T[i] * u_xx[i])

    #     else:
    #         k_m[i] = kramp(u_pred[i], k_l_t,k_s_t,T_L_tensor,T_S_tensor)
    #         cp_m[i] = cp_ramp(u_pred[i], cp_l_t,cp_s_t,T_L_tensor,T_S_tensor)
    #         rho_m[i] = rho_ramp(u_pred[i], rho_l_t,rho_s_t,T_L_tensor,T_S_tensor)
    #         # print(k_m.shape)
    #         alpha_T[i] = (k_m[i] / (rho_m[i] * (cp_m[i] + (L_fusion_t / (T_L_tensor - T_S_tensor)))))
    #         residual[i] = u_t[i] - (alpha_T[i] * u_xx[i])

    # # print(T_S_tensor,T_L_tensor,alpha_l_t,alpha_s_t)   
    # # print(u_pred[:5,:],alpha_T[:5,:],residual[:5,:])   
    
    
    
    # print(res_sq.dtype) 
    resid_mean = torch.mean(torch.square(residual)) 
    # print(resid_mean.dtype)

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
   

    return nn.MSELoss()(res_l,torch.zeros_like(res_l))

def ic_loss(u_pred,temp_init):
    temp_init_tsr = torch.tensor(temp_init,device=device)
   
    ic_mean = torch.mean(torch.square(u_pred -temp_init_tsr))
   
    return ic_mean

def accuracy(u_pred, u_true):
    return torch.mean(torch.abs(u_pred - u_true) / u_true)