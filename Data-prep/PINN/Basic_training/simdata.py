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

class HT_sim():
    # class to simulate heat transfer in a rod
    # the class has the following attributes like material properties, length of the rod, time of simulation, number of points, initial temperature, and surrounding temperature
    # the class has the following methods like dx_calc, dt_calc, cflcheck, step_coeff_calc, datagen, plot_temp
    

    def __init__(self, length, time_end, num_points, t_surr,temp_init):
        self.length = length
        self.time_end = time_end
        self.num_points = num_points
        self.t_surr = t_surr
        self.temp_init = temp_init
        self.dm = 60.0e-3 

        # Material properties
        self.rho = 2300.0                     # Density of AL380 (kg/m^3)
        self.rho_l = 2460.0                   # Density of AL380 (kg/m^3)
        self.rho_s = 2710.0                    # Density of AL380 (kg/m^3)
        self.rho_m = (self.rho_l + self.rho_s )/2       # Desnity in mushy zone is taken as average of liquid and solid density

        self.k = 104.0                       # W/m-K
        self.k_l = self.k                       # W/m-K
        self.k_s = 96.2                    # W/m-K
        self.k_m =  (self.k_l+self.k_s)/2                     # W/m-K
        self.k_mo = 41.5

        self.cp = 1245.3                      # Specific heat of aluminum (J/kg-K)
        self.cp_l = self.cp                      # Specific heat of aluminum (J/kg-K)
        self.cp_s = 963.0                 # Specific heat of aluminum (J/kg-K)
        self.cp_m =  (self.cp_l+self.cp_s)/2                 # Specific heat of mushy zone is taken as average of liquid and solid specific heat

        self.alpha_l = self.k_l / (self.rho_l * self.cp_l)
        self.alpha_s = self.k_s / (self.rho_s*self.cp_s)
        self.alpha_m = self.k_m / (self.rho_m * self.cp_m)          #`Thermal diffusivity in mushy zone is taken as average of liquid and solid thermal diffusivity`
        print(self.alpha_m, self.alpha_l, self.alpha_s)
        self.L_fusion = 389.0e3               # J/kg  # Latent heat of fusion of aluminum
        self.T_L = 574.4 +273.0                       #  K -Liquidus Temperature (615 c) AL 380
        self.T_S = 497.3 +273.0                     # K- Solidus Temperature (550 C)
        self.m_eff =(self.k_m/(self.rho_m*(self.cp_m + (self.L_fusion/(self.T_L-self.T_S)))))

        # sim field generation
        self.tempfield = np.full(self.num_points, self.temp_init)            # Initial temperature of the rod with ghost points at both ends
                         # Initial temperature of the rod

                                # Index of the midpoint
                  # List to store temperature at midpoint over time
                                                                   # die thickness in m
        # Calculate dx,dt, and step_coeff
        self.dx = self.dx_calc(self.length, self.num_points)
        print("The spatial step is", self.dx)
        self.dt = self.dt_calc(self.dx, self.alpha_l, self.alpha_s, self.alpha_m)
        print("The time step is", self.dt)
        self.step_coeff = self.step_coeff_calc(self.dt, self.dx)
        self.num_steps = round(self.time_end/self.dt)


    

    def dx_calc(self,length, num_points):
        dx = length / (num_points - 1)
        return dx
    
    def dt_calc(self,dx, alpha_l, alpha_s, alpha_m):
        maxi = max(alpha_s,alpha_l,alpha_m)
        dt = abs(0.5*((dx**2) /maxi))
        return dt
    def cflcheck(self,dx, alpha_l, alpha_s, alpha_m):
        cfl = 0.5 *(dx**2/max(alpha_l,alpha_s,alpha_m))
        if cfl > 1:
            print("CFL condition not satisfied")
            sys.exit()
        else:
            print("CFL condition satisfied")
        
        return cfl
    def step_coeff_calc(self,dt, dx):
        step_coeff = dt / (dx ** 2)
        return step_coeff


    def datagen(self):

        tempfield = self.tempfield.copy() 
       
        temp_int = self.tempfield.copy()
        self.temphist = [tempfield.copy()]
        
        for m in range(1, self.num_steps+1):                                                                            # time loop
            
            tempfield[0] = self.t_surr
            tempfield[-1] = self.t_surr

            for n in range(1,self.num_points-1):              # space loop, adjusted range
                tempfield[n] += ((self.alpha_l * self.step_coeff) * (temp_int[n+1] - (2.0 * temp_int[n]) + temp_int[n-1]))
                # phase[n] = 0
                                                                            # Update temperature
            temp_int = tempfield.copy()                                                                  # Update last time step temperature
            self.temphist.append(tempfield.copy())                                                  # Append the temperature history to add ghost points
                                            # Store midpoint temperature
        
        self.temp_history_1 = np.array(self.temphist)
        
        return self.temp_history_1
    
    def plot_temp(self,idx):
        # Plot the temperature distribution over time at the midpoint
        time_ss= np.linspace(0, self.time_end, self.num_steps+1)
        dx = self.dx
        plt.figure(figsize=(10, 6))
        plt.plot(time_ss, self.temp_history_1[:,idx], label='Midpoint Temperature')
        # plt.axhline(y=self.T_L, color='r', linestyle='--', label='Liquidus Temperature')
        # plt.axhline(y=self.T_S, color='g', linestyle='--', label='Solidus Temperature')
        plt.xlabel('Time(s)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature Distribution Over Time at x = {idx*dx*1000:.2f} mm') 
        plt.legend()
        plt.show()

 
                                                             # Update temperature

def fdd(length, time_end, num_points, num_steps, scl="True"):
    # module to create finite difference data
    x = np.linspace(0, length, num_points)
   
    t = np.linspace(0, time_end, num_steps)
    X, T = np.meshgrid(x, t)
    x = X.flatten()
    t = T.flatten()
    # print(x)
    # if scl == "True":
    #     x = scaler(x, 0, length)
    #     t = scaler(t, 0, time_end) 
    inp_fdd = np.column_stack((x, t))
    return inp_fdd

def quasirandom(n_samples, sampler, x_min,x_max, t_min, t_max):
    # module to create quasi-random sampling for pde input data
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
    # module to create uniform sampling and random sampling for pde input data
    if sampler == "random":
        x = np.random.uniform(x_min, x_max, n_samples)
        t = np.random.uniform(t_min, t_max, n_samples)
        inp = np.column_stack((x, t))
    elif sampler == "uniform":
        x = np.linspace(x_min, x_max, n_samples)
        t = np.linspace(t_min, t_max, n_samples)
        inp = np.column_stack((x, t))
    return inp





def pdeinp(x_min, x_max, t_min, t_max, n_samples, sampler, scl="True"):
     
    # module to create PDE inputs with various sampling strategies
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
    if scl=="True":
        print("scaling initated")
        inp_pde[:,0] = scaler(inp_pde[:,0], x_min, x_max)
        inp_pde[:,1] = scaler(inp_pde[:,1], t_min, t_max)
    else:
        print("scaling not initiated") 
    print("The number of points in the PDE input is", len(inp_pde))
    return inp_pde

    #sample the data between input and out

    #meshgrid the same

    #flatten the meshgrid and return the output

def icinp(length, icpts,scl="True"):
    # module to create initial condition inputs
    x = np.linspace(0, length, icpts)
    t= np.zeros(len(x))
    print("The number of points in the initial condition is", len(x))
    if scl == "True":
        x = scaler(x, 0, length)
    else:
        print("scaling not initiated")
    inp_ic = np.column_stack((x, t))
    return inp_ic

def bcinp(length, time_end, bcpts, delt, scl="True"):
    # module to create boundary condition inputs
    x_l = np.zeros(bcpts)
    x_r = np.ones(bcpts)*length

    t = np.linspace(0+delt, time_end, bcpts)
    print("The number of points in the left boundary condition is", len(x_l))
    print("The number of points in the right boundary condition is", len(x_r))

    if scl == "True":
        x_l = scaler(x_l, 0, length)
        x_r = scaler(x_r, 0, length)
        t = scaler(t, 0, time_end)
    else:
        print("scaling not initiated")
    inp_bcl = np.column_stack((x_l, t))
    inp_bcr = np.column_stack((x_r, t))
    return inp_bcl, inp_bcr
    

def scaler(data, min, max):
    # Scale the data between 0 and 1
    scaled_data = (data-min)/(max-min)
    return scaled_data

def invscaler(data, min, max):
    # Inverse scaling to bring the data back to original scale
    invsc_data = data*(max-min) + min
    return invsc_data

