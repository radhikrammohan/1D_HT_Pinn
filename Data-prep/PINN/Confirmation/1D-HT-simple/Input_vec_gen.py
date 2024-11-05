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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch


def input_gen(space,time, type):
    if type == 'scr': # this to create a space-time for initial and boundary conditions
        scaler1 = StandardScaler() # Standardize the input
        scaler2 = StandardScaler() # Standardize the input
        space_tr = scaler1.fit_transform(space.reshape(-1,1)) # Reshape the input to 2D array
        time_tr = scaler2.fit_transform(time.reshape(-1,1)) # Reshape the input to 2D array
        input_vec = np.hstack((space_tr,time_tr)) # Stack the input arrays horizontally
    if type == 'mgrid':  # this to create a grid for the entire space-time domain

        scaler3 = StandardScaler()
        scaler4 = StandardScaler()

        space_tr = scaler3.fit_transform(space.reshape(-1,1))
        time_tr = scaler4.fit_transform(time.reshape(-1,1))

        space_tr, time_tr = np.meshgrid(space_tr, time_tr)
        space_tr = space_tr.flatten().reshape(-1,1)
        time_tr = time_tr.flatten().reshape(-1,1)

        input_vec = np.hstack((space_tr,time_tr))

    return input_vec

