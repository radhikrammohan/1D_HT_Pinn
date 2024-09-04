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


def input_gen(space,time, type):
    if type == 'scr': # this to create a space-time for initial and boundary conditions
        scaler = StandardScaler() # Standardize the input
        space_tr = scaler.fit_transform(space.reshape(-1,1)) # Reshape the input to 2D array
        time_tr = scaler.fit_transform(time.reshape(-1,1)) # Reshape the input to 2D array
        input_vec = np.hstack((space_tr,time_tr)) # Stack the input arrays horizontally
    if type == 'mgrid':  # this to create a grid for the entire space-time domain

        scaler = StandardScaler()

        space_tr = scaler.fit_transform(space.reshape(-1,1))
        time_tr = scaler.fit_transform(time.reshape(-1,1))

        space_tr, time_tr = np.meshgrid(space_tr, time_tr)
        space_tr = space_tr.flatten().reshape(-1,1)
        time_tr = time_tr.flatten().reshape(-1,1)

        input_vec = np.hstack((space_tr,time_tr))

    return input_vec

def temp_data_gen(Temp,space,time):
    # Split the Temp into pde, ic, bc
    Temp_pde = Temp[1:,1:-1]
    Temp_ic = Temp[0,:]
    Temp_bc_l = Temp[:,0]
    Temp_bc_r = Temp[:,-1]

    

    
    return Temp_pde,Temp_ic,Temp_bc_l,Temp_bc_r,\
           
def st_gen(space,time):
    
    space, time = np.meshgrid(space, time)
    
    space_pde = space[1:,1:-1]
    time_pde = time[1:,1:-1]

    space_ic = space[0,:]
    time_ic = time[0,:]

    space_bc_l = space[:,0]
    time_bc_l = time[:,0]

    space_bc_r = space[:,-1]
    time_bc_r = time[:,-1]
    return space_pde, space_ic, space_bc_l, space_bc_r \
              ,time_pde, time_ic, time_bc_l, time_bc_r

def meshgen(space,time):
    space = space.flatten()
    time = time.flatten()


    return space, time

def input_3gen(space,time,htc):
    input = np.column_stack((space,time,htc))
    return input

