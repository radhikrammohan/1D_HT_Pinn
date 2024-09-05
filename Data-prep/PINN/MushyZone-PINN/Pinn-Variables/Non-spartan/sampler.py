import numpy as np

def g_sampler (x,t,samp_size):
    index = np.random.choice(x.shape[0],samp_size,replace=False)
    return x[index] , t[index]

