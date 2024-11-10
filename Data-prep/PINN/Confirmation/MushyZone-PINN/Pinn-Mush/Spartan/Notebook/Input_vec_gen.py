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
    if type == 'scr':
        scaler = StandardScaler()
        space_tr = scaler.fit_transform(space.reshape(-1,1))
        time_tr = scaler.fit_transform(time.reshape(-1,1))
        input_vec = np.hstack((space_tr,time_tr))
    if type == 'mgrid':

        scaler = StandardScaler()

        space_tr = scaler.fit_transform(space.reshape(-1,1))
        time_tr = scaler.fit_transform(time.reshape(-1,1))

        space_tr, time_tr = np.meshgrid(space_tr, time_tr)
        space_tr = space_tr.flatten().reshape(-1,1)
        time_tr = time_tr.flatten().reshape(-1,1)

        input_vec = np.hstack((space_tr,time_tr))

    return input_vec

