import numpy as np
import matplotlib.pyplot as plt

def sim1d(rho_l, rho_s, k_l, k_s, cp_l, cp_s,t_surr, L_fusion, temp_init,htc_l,htc_r):
    

    # Geometry
    length = 15e-3                    # Length of the rod
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
        time_end = time_end +dt                                                 # Update end time
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
      
    temperature_history_1 = np.array(temperature_history)                       # Convert temperature history to numpy array
    phi_history_1 = np.array(phi_history)                                      # Convert phase history to numpy array
    aa = np.array(temperature_history)
    ab = np.array(phi_history)
    temp_hist_l = aa[:,1:-1]
    phi_history_1 = ab[:,1:-1]
    print(temp_hist_l.shape)
    t_dim,x_dim  = temp_hist_l.shape
    # Niyama Calcualtion

    # print(temperature_history_1.shape)
    # Gradient Calculation

    grad_t_x = np.absolute(np.gradient(temperature_history_1, dx, axis=1))     # Gradient of temperature with respect to space
    # print(grad_t_x[100,:])
    grad_t_t = np.absolute(np.gradient(temperature_history_1,dt,axis=0))       # Gradient of temperature with respect to time
    # print(grad_t_t[100,:])
    sq_grad_t_t = np.square(grad_t_t)                                         # Square of the gradient of temperature with respect to space
    Ny = np.divide(grad_t_x, sq_grad_t_t, out=np.zeros_like(grad_t_x, dtype=float), where=sq_grad_t_t!=0)         # Niyama number
    # print(Ny)


    C_lambda = 40.0e-06                                                                 # C Lambda for Niyama number calculation
    del_Pcr = 1.01e5                                                                   # Critical Pressure difference
    dyn_visc = 1.2e-3                                                                  # Dynamic Viscosity
    beta = (rho_s - rho_l)/ rho_l                                                     # Beta    
    # print(beta)
    del_Tf = T_L - T_S                                                               # Delta T
    # print(del_Tf)
    k1a=(dyn_visc*beta*del_Tf)                                                      
    k1 = (del_Pcr/k1a)**(1/2)
    # print(k1)
    num_steps = temp_hist_l.shape[0]-1                                # Number of time steps
    # print(num_steps)

    # k2 = np.divide(grad_t_x, grad_t_t_power, out=np.zeros_like(grad_t_x, dtype=float), where=grad_t_t_power!=0)
    k2 = np.zeros((num_steps+1,num_points))
    k3 = np.zeros((num_steps+1,num_points))
    for i in range(num_steps+1):
        for j in range(num_points):
            if grad_t_x[i,j] == 0:
                k2[i,j] = 0
                k3[i,j] = 0
            if grad_t_t[i,j]== 0:
                k2[i,j] = 0
                k3[i,j] = 0
            else:
                k2[i,j] = ((grad_t_x[i,j]))/ (((grad_t_t[i,j]))**(5/6))
                k3[i,j] = (grad_t_x[i,j])/ ((grad_t_t[i,j])**(1/2))
        
    # k2 = grad_t_x/((grad_t_t)**(5/6))
    # print(k2)
    Ny_s= k3
    Dim_ny = C_lambda * k1 * k2
    # print(Dim_ny)

    # print(grad_t_t[:, 50])
    # plot = plt.figure(figsize=(10, 6))
    # plt.plot(time_ss, grad_t_x[:, 50], label='Niyama Number at x = 7.5mm')
    # plt.xlabel('Time(s)')
    # plt.ylabel('Niyama Number')
    # plt.title('Niyama Number Distribution Over Time at x = 7.5mm')
    # plt.legend()
    # plt.show()**
    
    
    Ny_time = 0.90*current_time                                     # Time at which Niyama number is calculated 

    Ny_index = int(Ny_time/dt)                                      # Index of the time at which Niyama number is calculated
    Cr_Ny = np.min(Dim_ny[Ny_index, :])
    Cr_Nys = np.min(Ny_s[Ny_index,:])                                # Minimum Niyama number at the time of interest
    
    indices =[]
    threshold = T_S + 0.1*(T_L-T_S)
    tolerance = 1.0
    # print(threshold)

    for i in range (t_dim):
        for j in range(x_dim):
            if np.absolute(temp_hist_l[i,j]- threshold) < tolerance:
                indices.append((i,j))

    # print(indices)

    

    # print(Dim_ny.shape)
    Niyama_pct = [Dim_ny[i,j] for i,j in indices]
    Niyama_array = np.array(Niyama_pct)
    Lowest_Niyama = np.min(Niyama_array)
    Avg_Niyama = np.mean(Niyama_array)

 


# # Check the new shape after transposing
# print("Transposed Temperature History Shape:", temperature_history.shape)
# print("Transposed Phi History Shape:", phi_history.shape)

    # Create a meshgrid for space and time coordinates
    space_coord, time_coord = np.meshgrid(np.arange(temp_hist_l.shape[1]), np.arange(temp_hist_l.shape[0]))

    time_coord = time_coord * dt 

    hlt_t, hlt_x = zip(*indices)
    real_t = []

    for index in indices:
        real_t.append(time_coord[index[0],index[1]])

    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

    # Plot the temperature history on the left subplot
    im1 = ax1.pcolormesh(space_coord, time_coord, temp_hist_l, cmap='viridis')
    ax1.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=16)
    ax1.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
    ax1.set_title('Temperature Variation Over Time',fontname='Times New Roman', fontsize=20)
    fig.colorbar(im1, ax=ax1, label='Temperature')

    # Plot the phase history on the right subplot
    im2 = ax2.pcolormesh(space_coord, time_coord, phi_history_1, cmap='viridis')
    ax2.set_xlabel('Space Coordinate', fontname='Times New Roman', fontsize=18)
    ax2.set_ylabel('Time',fontname='Times New Roman', fontsize=16)
    ax2.set_title('Phase Variation Over Time',fontname='Times New Roman', fontsize=20)
    fig.colorbar(im2, ax=ax2, label='Phase')


    #plot the main
    # fig, ax = plt.subplots(figsize=(14, 6))

    im3 = ax3.pcolormesh(space_coord, time_coord, Dim_ny[:,1:-1], cmap='viridis')
    im3 = ax3.scatter(hlt_x, real_t, color='r', s=1, zorder=5, label='Highlighted Points')  # s=100 sets the marker size, zorder=5 puts the points on top
    ax3.set_xlabel('Space Coordinate')
    ax3.set_ylabel('Time')
    ax3.set_title('Niyama Variation Over Time')
    fig.colorbar(im3, ax= ax3, label='Dimesnionless Niyama Number')



    plt.tight_layout()
    plt.show()




    print(f'Lowest Niyama:{Lowest_Niyama}, rho_l:{rho_l}, rho_s:{rho_s}, k_l:{k_l}, k_s:{k_s}, cp_l:{cp_l}, cp_s:{cp_s}, t_surr:{t_surr}, L_fusion:{L_fusion}, temp_init:{temp_init},htc_l:{htc_l},htc_r:{htc_r}')
    return current_time, temperature_history, phi_history, Cr_Ny,Cr_Nys, Lowest_Niyama, Avg_Niyama



