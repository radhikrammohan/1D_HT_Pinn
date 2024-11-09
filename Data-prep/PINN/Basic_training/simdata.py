# The module calcualtes the heat transfer equation in one phase and give a temperature array as output.
# The module also gives a figure of the distribution as output. 
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skopt
from distutils.version import LooseVersion
import csv

# Geometry


# Material properties
rho = 2300.0                     # Density of AL380 (kg/m^3)
rho_l = 2460.0                   # Density of AL380 (kg/m^3)
rho_s = 2710.0                    # Density of AL380 (kg/m^3)
rho_m = (rho_l + rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

k = 104.0                       # W/m-K
k_l = k                       # W/m-K
k_s = 96.2                    # W/m-K
k_m =  (k_l+k_s)/2                     # W/m-K
k_mo = 41.5


cp = 1245.3                      # Specific heat of aluminum (J/kg-K)
cp_l = cp                      # Specific heat of aluminum (J/kg-K)
cp_s = 963.0                 # Specific heat of aluminum (J/kg-K)
cp_m =  (cp_l+cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat
# cp_m = cp
           # Thermal diffusivity
alpha_l = k_l / (rho_l * cp_l) 
alpha_s = k_s / (rho_s*cp_s)
alpha_m = k_m / (rho_m * cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`


#L_fusion = 3.9e3                 # J/kg
L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum
         # Thermal diffusivity


T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380
T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)
m_eff =(k_m/(rho_m*(cp_m + (L_fusion/(T_L-T_S)))))
# print (f"alpha_l = {alpha_l}, alpha_s = {alpha_s}, m_eff = {m_eff}")

# htc = 10.0                   # W/m^2-K
# q = htc*(919.0-723.0)
# q = 10000.0






                               # Surrounding temperature in K
# t_surr = h()



def kramp(temp,v1,v2,T_L,T_s):                                      # Function to calculate thermal conductivity in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        if temp > T_L:
            k_m = k_l
        elif temp < T_S:
            k_m = k_s
        else:
            k_m = k_s + slope*(temp-T_S)
        return k_m

def cp_ramp(temp,v1,v2,T_L,T_s):                                    # Function to calculate specific heat capacity in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    if temp > T_L:
        cp_m = cp_l
    elif temp < T_S:
        cp_m = cp_s
    else:
        cp_m = cp_s + slope*(temp-T_S)
    return cp_m

def rho_ramp(temp,v1,v2,T_L,T_s):                                       # Function to calculate density in Mushy Zone
    slope = (v1-v2)/(T_L-T_S)
    if temp > T_L:
        rho_m = rho_l
    elif temp < T_S:
        rho_m = rho_s
    else:
        rho_m = rho_s + slope*(temp-T_S)
    return rho_m

def datagen(temp_init, t_surr, numpoints, len, time_end):

    # spatial discretization

    num_points = numpoints                        # Number of spatial points
    dx = len / (num_points - 1)         # Distance between two spatial points
    print('dx is',dx)

    # Time Discretization  
    # 
    time_end = time_end        # seconds                         
    maxi = max(alpha_s,alpha_l,alpha_m)
    dt = abs(0.5*((dx**2) /maxi)) 

    print('dt is ',dt)
    num_steps = round(time_end/dt)
    print('num_steps is',num_steps)

    # Stability Condition Checking
    cfl = 0.5 *(dx**2/max(alpha_l,alpha_s,alpha_m))
    print('cfl is',cfl)

    time_steps = np.linspace(0, time_end, num_steps + 1)
    step_coeff = dt / (dx ** 2)

    if dt <= cfl:
        print('stability criteria satisfied')
    else:
        print('stability criteria not satisfied')
        sys.exit()
    
    # temp field init

    temp_in = temp_init  # This can be in function calls
    # Initial temperature and phase fields
    temperature = np.full(num_points+2, temp_in)            # Initial temperature of the rod with ghost points at both ends
    # phase = np.zeros(num_points+2)*0.0                    # Initial phase of the rod with ghost points at both ends

    # Set boundary conditions
    # temperature[-1] = 919.0 
    # phase[-1] = 1.0

    # temperature[0] = 919.0 #(40 C)
    # phase[0] = 1.0

    # Store initial state in history
    temperature_history = [temperature.copy()]    # List to store temperature at each time step
    # phi_history = [phase.copy()]                    # List to store phase at each time step
    temp_int = temperature.copy()                 # Initial temperature of the rod
    # print(temperature_history,phi_history)
    # Array to store temperature at midpoint over time
    midpoint_index = num_points // 2                          # Index of the midpoint 

    midpoint_temperature_history = [temperature[midpoint_index]]            # List to store temperature at midpoint over time
    dm = 60.0e-3                                                            # die thickness in m

    # r_m =  (k_mo / dm) + (1/htc)
    
    t_surr = t_surr        
    
    
    
    
    for m in range(1, num_steps+1):                                                                            # time loop
        # htc = 10.0                   # htc of Still air in W/m^2-K
        # q1 = htc*(temp_int[0]-t_surr)   # Heat flux at the left boundary 
        
        # print(f"q1 is {q1}")
        temperature[0] = t_surr # Update boundary condition temperature
        
        # q2 = htc*(temp_int[-1]-t_surr)                   # Heat flux at the right boundary
        temperature[-1] = t_surr  # Update boundary condition temperature
        
        for n in range(1,num_points+1):              # space loop, adjusted range
        
            temperature[n] += ((alpha_l * step_coeff) * (temp_int[n+1] - (2.0 * temp_int[n]) + temp_int[n-1]))
            # phase[n] = 0
             
          
        temperature = temperature.copy()                                                                # Update temperature
        # phase = phase.copy()                                                                            # Update phase
        temp_int = temperature.copy()                                                                  # Update last time step temperature
        temperature_history.append(temperature.copy())                                                  # Append the temperature history to add ghost points
        # phi_history.append(phase.copy())                                                                # Append the phase history to add ghost points
        midpoint_temperature_history.append(temperature[midpoint_index])                                # Store midpoint temperature
        
    
    # print(f"Step {m}, Temperature: {temperature}")
    


# print(midpoint_temperature_history)
#print(phi_history)



# Plot temperature history for debugging
    temperature_history_1 = np.array(temperature_history)
    print(temperature_history_1.shape)
    time_ss= np.linspace(0, time_end, num_steps+1)
    # print(time_ss.shape)
    plt.figure(figsize=(10, 6))
    plt.plot(time_ss, midpoint_temperature_history, label='Midpoint Temperature')
    plt.axhline(y=T_L, color='r', linestyle='--', label='Liquidus Temperature')
    plt.axhline(y=T_S, color='g', linestyle='--', label='Solidus Temperature')
    plt.xlabel('Time(s)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Distribution Over Time at x = 7.5mm') 
    plt.legend()
    plt.show()

    return temperature_history_1

                                                               # Update temperature

def fdd(length, time_end, num_points, num_steps):
    x = np.linspace(0, length, num_points)
    t = np.linspace(0, time_end, num_steps)
    X, T = np.meshgrid(x, t)
    x = X.flatten()
    t = T.flatten()
    inp_fdd = np.column_stack((x, t))
    return inp_fdd

def quasirandom(n_samples, sampler, x_min,x_max, t_min, t_max):
    space = [(x_min, x_max), (t_min, t_max)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(
                sampler.generate(space, n_samples + 2)[2:]
            )
    return np.array(sampler.generate(space, n_samples))

def unidata(x_min, x_max, t_min, t_max, n_samples, sampler):

    if sampler == "random":
        x = np.random.uniform(x_min, x_max, n_samples)
        t = np.random.uniform(t_min, t_max, n_samples)
        inp = np.column_stack((x, t))
    elif sampler == "uniform":
        x = np.linspace(x_min, x_max, n_samples)
        t = np.linspace(t_min, t_max, n_samples)
        inp = np.column_stack((x, t))
    return inp





def pdeinp(x_min, x_max, t_min, t_max, n_samples, sampler):
     
    
    # define a sampling strategy
    if sampler == "random":
        inp_pde = unidata(x_min, x_max, t_min, t_max, n_samples, sampler)
    elif sampler == "uniform":
        inp_pde = unidata(x_min, x_max, t_min, t_max, n_samples, sampler)
    elif sampler == "LHS":
        inp_pde = quasirandom(n_samples, "LHS", x_min, x_max, t_min, t_max)
    elif sampler == "Halton":
        inp_pde = quasirandom(n_samples, "Halton", x_min, x_max, t_min, t_max)
    elif sampler == "Hammersley":
        inp_pde = quasirandom(n_samples, "Hammersley", x_min, x_max, t_min, t_max)
    elif sampler == "Sobol":
        inp_pde = quasirandom(n_samples, "Sobol", x_min, x_max, t_min, t_max)
    else:
        raise ValueError("Invalid sampler specified. Choose from 'random', 'uniform', 'LHS', 'Halton', 'Hammersley', 'Sobol'.")
    return inp_pde

    #sample the data between input and out

    #meshgrid the same

    #flatten the meshgrid and return the output

def icinp(length, icpts):

    x = np.linspace(0, length, icpts)
    t= np.zeros(len(x))
    inp_ic = np.column_stack((x, t))
    return inp_ic

def bcinp(length, time_end, bcpts):

    x_l = np.zeros(bcpts)
    x_r = np.ones(bcpts)*length

    t = np.linspace(0, time_end, bcpts)
    inp_bcl = np.column_stack((x_l, t))
    inp_bcr = np.column_stack((x_r, t))
    return inp_bcl, inp_bcr
    