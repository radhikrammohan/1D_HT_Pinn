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


def input_gen(space,time, type,scale = False):
    if type == 'scr': # this to create a space-time for initial and boundary conditions
        if scale:
            scaler1 = StandardScaler()
            scaler2 = StandardScaler()
            space = scaler1.fit_transform(space.reshape(-1,1))
            time = scaler2.fit_transform(time.reshape(-1,1))
            input_vec = np.hstack((space,time))
        else:
            input_vec = np.hstack((space,time))
        
    if type == 'mgrid':  # this to create a grid for the entire space-time domain
        if scale:
            scaler3 = StandardScaler()
            scaler4 = StandardScaler()
            space_ = scaler3.fit_transform(space.reshape(-1,1))
            time = scaler4.fit_transform(time.reshape(-1,1))
            space, time = np.meshgrid(space, time)
            space = space.flatten().reshape(-1,1)
            time = time.flatten().reshape(-1,1)
            input_vec = np.hstack((space,time))

        else:
            space, time = np.meshgrid(space, time)
            space = space.flatten().reshape(-1,1)
            time = time.flatten().reshape(-1,1)
            input_vec = np.hstack((space.flatten().reshape(-1,1),time.flatten().reshape(-1,1)))

    return input_vec ,space, time

def temp_data_gen(Temp):
    # Split the Temp into pde, ic, bc
    
    Temp_pde = Temp[1:,1:-1]
    Temp_pde = Temp_pde.flatten().reshape(-1,1)
    Temp_ic = Temp[0,:]
    Temp_ic = Temp_ic.flatten().reshape(-1,1)
    Temp_bc_l = Temp[:,0]
    Temp_bc_l = Temp_bc_l.flatten().reshape(-1,1)   
    Temp_bc_r = Temp[:,-1]
    Temp_bc_r = Temp_bc_r.flatten().reshape(-1,1)
    Temp = Temp.flatten().reshape(-1,1)

    
    return Temp,Temp_pde,Temp_ic,Temp_bc_l,Temp_bc_r,
           
def st_gen(space,time):
    
    space, time = np.meshgrid(space, time)
    
    space_pde = space[1:,1:-1]
    space_pde = space_pde.flatten().reshape(-1,1)
    time_pde = time[1:,1:-1]
    time_pde = time_pde.flatten().reshape(-1,1)

    space_ic = space[0,:]
    space_ic = space_ic.flatten().reshape(-1,1)
    time_ic = time[0,:]
    time_ic = time_ic.flatten().reshape(-1,1)

    space_bc_l = space[:,0]
    space_bc_l = space_bc_l.flatten().reshape(-1,1)
    time_bc_l = time[:,0]
    time_bc_l = time_bc_l.flatten().reshape(-1,1)

    space_bc_r = space[:,-1]
    time_bc_r = time[:,-1]
    return space_pde, space_ic, space_bc_l, space_bc_r \
              ,time_pde, time_ic, time_bc_l, time_bc_r

def meshgen(space,time):
    space = space.flatten()
    time = time.flatten()


    return space, time

def input_vgen(space,time):
    input = np.column_stack((space,time))
    return input

