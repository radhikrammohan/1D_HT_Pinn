import numpy as np

def g_sampler (x,t,samp_size):

    l1 = len(x)
    l2 = len(t)
    index = np.random.choice(x.shape[0],samp_size,replace=False)
    return x[index] , t[index]

def strat_sampler (x,t,percent):

    l1 = len(x)
    indices_1 = list(range(l1))

    l2 = len(t)
    indices_2 = list(range(l2))

    bins_l1 = l1 //3
    bins_l2 = l2 //3

    l1_indices_1 = indices_1[:bins_l1] 
    l1_indices_2 = indices_1[bins_l1:2*bins_l1]
    l1_indices_3 = indices_1[2*bins_l1:]

    l2_indices_1 = indices_2[:bins_l2]
    l2_indices_2 = indices_2[bins_l2:2*bins_l2]
    l2_indices_3 = indices_2[2*bins_l2:]

    bin1 = np.random.choice(l1_indices_1, int(percent*bins_l1), replace=False)
    bin2 = np.random.choice(l1_indices_2, int(percent*bins_l1), replace=False)
    bin3 = np.random.choice(l1_indices_3, int(percent*bins_l1), replace=False)


    bin4 = np.random.choice(l2_indices_1, int(percent*bins_l2), replace=False)
    bin5 = np.random.choice(l2_indices_2, int(percent*bins_l2), replace=False)
    bin6 = np.random.choice(l2_indices_3, int(percent*bins_l2), replace=False)

    index1 = np.concatenate((bin1,bin2,bin3))
    index2 = np.concatenate((bin4,bin5,bin6))

    return x[index1] , t[index2]
    