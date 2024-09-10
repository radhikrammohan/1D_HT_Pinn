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


def input_gen(space,time, type,scale):
    
    if type == 'scr': # this to create a space-time for initial and boundary conditions
        if scale == 1:
            scaler = StandardScaler()
            space_tr = scaler.fit_transform(space.reshape(-1,1)) # Reshape the input to 2D array
            time_tr = scaler.fit_transform(time.reshape(-1,1)) # Reshape the input to 2D array
            input_vec = np.hstack((space_tr,time_tr)) # Stack the input arrays horizontally
        else:
            space_tr = space.reshape(-1,1) # Reshape the input to 2D array
            time_tr = time.reshape(-1,1) # Reshape the input to 2D array
            input_vec = np.hstack((space_tr,time_tr)) # Stack the input arrays horizontally
    
    if type == 'mgrid':  # this to create a grid for the entire space-time domain
        if scale == 1:
            scaler = StandardScaler()
            space_tr = space.reshape(-1,1)
            time_tr = time.reshape(-1,1)
            space_tr, time_tr = np.meshgrid(space_tr, time_tr)
            space_tr = space_tr.flatten().reshape(-1,1)
            time_tr = time_tr.flatten().reshape(-1,1)
            input_vec = np.hstack((space_tr,time_tr)) # Stack the input arrays horizontally
        else:
            space_tr = space.reshape(-1,1)
            time_tr = time.reshape(-1,1)
            space_tr, time_tr = np.meshgrid(space_tr, time_tr)
            space_tr = space_tr.flatten().reshape(-1,1)
            time_tr = time_tr.flatten().reshape(-1,1)
            input_vec = np.hstack((space_tr,time_tr))

    return input_vec

