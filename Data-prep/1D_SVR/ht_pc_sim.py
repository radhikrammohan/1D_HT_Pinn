import numpy as np

def sim1d(T_L, T_S,rho_l, rho_s, rho_m, k_l, k_s, k_m,\
         cp_l, cp_s, cp_m, \
         L_fusion, \
          bc_l, bc_r, temp_init):
    

    # Geometry
    length = 1e-2                 # Length of the rod

    alpha_l = k_l / (rho_l * cp_l)  # Thermal diffusivity of the liquid
    alpha_s = k_s / (rho_s * cp_s)  # Thermal diffusivity of the solid
    alpha_m = k_m / (rho_m * cp_m)  # Thermal diffusivity of the solid
    # Spatial discretization

    num_points = 50                  # Number of spatial points
    dx = length / (num_points - 1)
    print('dx is',dx)
                                   #dt = time_end/num_steps
    #num_steps = 200000               # Number of time steps
                                  # num_steps = round(time_end/dt)
                                                              
    # Time Discretization  
    #time_end = 5                 # seconds                         
    #num_steps = 10000
    # dt = time_end/num_steps
    dt = abs(0.5 *(dx**2/(max(alpha_l, alpha_s, alpha_m))))
    
    num_steps = round(time_end/dt) +1
    
    
   
    #dt = time_end / num_steps
    time_steps = np.linspace(0, time_end, num_steps + 1)
       
    
        
    


    # Initial temperature and phase fields
    temperature = np.full(num_points, temp_init)
    phase = np.zeros(num_points)*1.0

    # Set boundary conditions
    temperature[-1] = bc_r #(40 C)
    phase[-1] = 1.0

    temperature[0] = bc_l #(40 C)
    phase[0] = 1.0

    # Store initial state in history
    temperature_history = [temperature.copy()]
    phi_history = [phase.copy()]
    
    
    
    for m in range(1, num_steps+1):                  # time loop
        for n in range(1,num_points-1):              # space loop, adjusted range
        #print(f"Step {m}, point {n},Temperature: {temperature}, Phase: {phase}")
            if temperature[n] >= T_L:
                temperature[n] = temperature[n] + ((alpha_l * dt )/ dx**2) * (temperature[n+1] - 2.0 * temperature[n] + temperature[n-1])
                phase[n] = 0
         
                #print(m,n,temperature[n],phase[n])
            elif T_S < temperature[n] < T_L:
            #temperature[n] = temperature[n] - (((k * dt) / (rho*(T_L-T_S)*(cp*(T_L-T_S)-L_fusion)*(dx**2))) * (temperature[n+1] - 2 * temperature[n] + temperature[n-1]))
                temperature[n] = temperature[n] - ((k_m/(rho_m*(cp_m-(L_fusion/(T_L-T_S)))))* (temperature[n+1] - 2 * temperature[n] + temperature[n-1]))
                phase[n] = (T_L - temperature[n]) / (T_L - T_S)
            #print(m,n,temperature[n],phase[n])
         
            elif temperature[n]<T_S:
                temperature[n] = temperature[n] + ((alpha_s * dt )/ dx**2) * (temperature[n+1] - 2.0 * temperature[n] + temperature[n-1])
                phase[n] = 1
            
            else:
                print("ERROR: should not be here")
         
           # print(m,n,temperature[n],phase[n])
    
        temperature_history.append(temperature.copy())
        phi_history.append(phase.copy())

    # Compute the average phase at the end of the simulation
    final_phase = phi_history[-1]
    
    average_solid_fraction = np.mean(final_phase)
    
    

    return average_solid_fraction
