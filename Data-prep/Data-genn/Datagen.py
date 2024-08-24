import numpy as np
import matplotlib.pyplot as plt

def sim1d(rho_l, rho_s, k_l, k_s, cp_l, cp_s,t_surr, L_fusion, temp_init,htc_l,htc_r, length, gen_graph=True, gen_data=True):
    

    # Geometry
    length = length                    # Length of the rod
    num_points = 50                   # Number of spatial points
    dx = length / (num_points - 1)    # Grid spacing

    # Material Properties

    def kramp(temp,v1,v2,T_L,T_s):   # Function to calculate thermal conductivity in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        if temp > T_L:
            k_m = k_l
        elif temp < T_S:
            k_m = k_s
        else:
            k_m = k_s + slope*(temp-T_S)
        return k_m

    def cp_ramp(temp,v1,v2,T_L,T_s):    # Function to calculate specific heat capacity in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        if temp > T_L:
            cp_m = cp_l
        elif temp < T_S:
            cp_m = cp_s
        else:
            cp_m = cp_s + slope*(temp-T_S)
        return cp_m

    def rho_ramp(temp,v1,v2,T_L,T_s):   # Function to calculate density in Mushy Zone
        slope = (v1-v2)/(T_L-T_S)
        if temp > T_L:
            rho_m = rho_l
        elif temp < T_S:
            rho_m = rho_s
        else:
            rho_m = rho_s + slope*(temp-T_S)
        return rho_m
    
    rho_l = rho_l                                  # Density of the liquid
    rho_s = rho_s                                  # Density of the solid
    
    k_l = k_l                                   # Thermal conductivity of the liquid
    k_s = k_s                                   # Thermal conductivity of the solid
    
    k_mo = 41.5                                   # Thermal conductivity of the mold

    cp_l = cp_l                                    # Specific heat capacity of the liquid
    cp_s = cp_s                                     # Specific heat capacity of the solid
    
    L_fusion = L_fusion                                # Latent heat of fusion

    T_L = 847.4                               #  K -Liquidus Temperature (615 c) AL 380
    T_S = 770.3                               # K- Solidus Temperature (550 C)                   # Solidus temperature
    
    alpha_l = k_l / (rho_l * cp_l)                       # Thermal diffusivity of the liquid
    alpha_s = k_s / (rho_s * cp_s)                      # Thermal diffusivity of the solid
     # Average thermal diffusivity
                                                                  
    # Time Discretization  
                  # seconds
    maxi =  max(alpha_l, alpha_s)                         
    dt = abs(0.5 *((dx**2)/maxi))                           # Time step
    step_coeff = dt/ (dx**2)
    current_time = dt  
    time_end = 1 
    
    # Initial temperature and phase fields
    temperature = np.full(num_points, temp_init)             # Initial temperature field 
    phase = np.zeros(num_points)*1.0                        # Initial phase field

    # Set boundary conditions
    # temperature[-1] = 723.0 #(40 C)
    phase[-1] = 1.0                                        # Set right boundary condition for phase                  

    # temperature[0] = 723.0 #(40 C)
    phase[0] = 1.0                                             # Set left boundary condition for phase
    

    # Store initial state in history
    temperature_history = [temperature.copy()]                    # Temperature history
    phi_history = [phase.copy()]                                  # Phase history
    temp_initf = temperature.copy()                                   # Additional temperature field for updating
    
    t_surr = t_surr                                              # Surrounding temperature             
    dm = 60.0e-3                                                  # thickness of the mold
    
    r_m = k_mo / dm                                                # Thermal Resistance of the mold

    # midpoint_index = num_points // 2

    # midpoint_temperature_history = [temperature[midpoint_index]]
    
    while current_time < time_end:  # time loop
        htc_l = htc_l
        htc_r = htc_r
        q1 = htc_l * (temp_initf[0] - t_surr)                     # Heat flux at the left boundary
        temperature[0] = temp_initf[0] + \
                        (alpha_l * step_coeff * \
                         ((2.0*temp_initf[1]) - \
                          (2.0 * temp_initf[0])-(2.0*dx*(q1))))  # Update left boundary condition temperature
        
        q2 = htc_r *(temp_initf[-1]-t_surr)                          # Heat flux at the right boundary
        temperature[-1] = temp_initf[-1] + \
                         (alpha_l * step_coeff \
                             * ((2.0*temp_initf[-2]) - \
                            (2.0 * temp_initf[-1])-(2.0*dx*(q2))))  # Update right boundary condition temperature               
        
        for n in range(1,num_points-1):              # space loop, adjusted range
                
            if temperature[n] >= T_L:
                temperature[n] += ((alpha_l * step_coeff) * (temp_initf[n+1] - (2.0 * temp_initf[n]) + temp_initf[n-1]))
                phase[n] = 0
                         
            elif T_S < temperature[n] < T_L:
            
                k_m = kramp(temperature[n],k_l,k_s,T_L,T_S)
                cp_m = cp_ramp(temperature[n],cp_l,cp_s,T_L,T_S)
                rho_m = rho_ramp(temperature[n],rho_l,rho_s,T_L,T_S)
                m_eff =(k_m/(rho_m*(cp_m + (L_fusion/(T_L-T_S)))))
                
                temperature[n] +=  ((m_eff * step_coeff)* (temp_initf[n+1] - (2.0 * temp_initf[n]) + temp_initf[n-1]))
                phase[n] = (T_L - temperature[n]) / (T_L - T_S) 
                  
            elif temperature[n] <= T_S:
                temperature[n] +=  ((alpha_s * step_coeff) * (temp_initf[n+1] - (2.0 * temp_initf[n])+ temp_initf[n-1]))
                phase[n] = 1
            
            else:
                print("ERROR: should not be here")
        
             
        current_time = current_time + dt                                         # Update current time
        time_end = time_end + dt                                             # Update end time
        
        temperature = temperature.copy()                                        # Update temperature field
        phase = phase.copy()                                                        # Update phase field
        temp_initf = temperature.copy()                                          # Update additional temperature field
        temperature_history.append(temperature.copy())                           # Store temperature field in history
        phi_history.append(phase.copy())                                             # Store phase field in history
        # midpoint_temperature_history.append(temperature[midpoint_index])        
        if np.all(phase == 1):
            # print("Simulation complete @ time: ", current_time)
            break
        
         # Check the new shape after transposing

    current_time = current_time - dt 
    num_steps = len(temperature_history) - 1
    
    temperature_history_1 = np.array(temperature_history)                       # Convert temperature history to numpy array
    phi_history_1 = np.array(phi_history)                                      # Convert phase history to numpy array
    aa = np.array(temperature_history)
    ab = np.array(phi_history)
    temp_hist_l = aa[:,1:-1]
    phi_history_1 = ab[:,1:-1]
    midpoint_temperature_history = temperature_history_1[:,num_points//2]

    t_hist = temp_hist_l
    p_hist = phi_history_1


    if gen_graph:
        # Create a meshgrid for space and time coordinates
        space_coord, time_coord = np.meshgrid(np.arange(t_hist.shape[1]), np.arange(t_hist.shape[0]))

        time_coord = time_coord * dt 
        # Create a figure with two subplots
        fig, (ax1) = plt.subplots(1,figsize=(14, 6))

        # Plot the temperature history on the left subplot
        im1 = ax1.pcolormesh(space_coord, time_coord, t_hist, cmap='coolwarm')
        ax1.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=16)
        ax1.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
        ax1.set_title('Temperature Variation Over Time',fontname='Times New Roman', fontsize=20)
        ax1.grid(True)
        fig.colorbar(im1, ax=ax1, label='Temperature')
        
        plt.contour(t_hist, colors='black', linewidths=0.5)
        
        plt.tight_layout()
        plt.show()
        # Plot temperature history for debugging
        temperature_history_1 = np.array(temperature_history)
        print(temperature_history_1.shape)
        time_ss= np.linspace(0, current_time, num_steps+1)
        # print(time_ss.shape)
        plt.figure(figsize=(10, 6))
        plt.plot(time_ss, midpoint_temperature_history, label='Midpoint Temperature')
        plt.axhline(y=T_L, color='r', linestyle='--', label='Liquidus Temperature')
        plt.axhline(y=T_S, color='g', linestyle='--', label='Solidus Temperature')
        plt.xlabel('Time(s)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Distribution Over Time at x = 7.5mm') 
        plt.legend()
        plt.grid(True)
        plt.show()
    if gen_data:
        return t_hist
    
  

    




    # print(f'Lowest Niyama:{Lowest_Niyama}, rho_l:{rho_l}, rho_s:{rho_s}, k_l:{k_l}, k_s:{k_s}, cp_l:{cp_l}, cp_s:{cp_s}, t_surr:{t_surr}, L_fusion:{L_fusion}, temp_init:{temp_init},htc_l:{htc_l},htc_r:{htc_r},length:{length}')
    



