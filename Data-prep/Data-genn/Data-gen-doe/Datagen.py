import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

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
    current_time = 0  
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
    
    
    Ny_time = 0.90 * current_time                                     # Time at which Niyama number is calculated 

    Ny_index = int(Ny_time/dt)                                      # Index of the time at which Niyama number is calculated
    Cr_Ny = np.min(Dim_ny[Ny_index, :])
    Cr_Nys = np.min(Ny_s[Ny_index,:])                                # Minimum Niyama number at the time of interest
    
    indices =[]
    indices_nim =[] # Indices of the Niyama number below threshold
    threshold = T_S + 0.1*(T_L-T_S)
    print(threshold)
    tolerance = 1.0
    # print(threshold)

    Dim_ny_new = np.copy(Dim_ny)

    for i in range(Dim_ny_new.shape[0]):
        for j in range(Dim_ny_new.shape[1]):
            if Dim_ny_new[i,j] > 3.5:
                Dim_ny_new[i,j] = 0
            else:
                Dim_ny_new[i,j] = Dim_ny_new[i,j]
    # print(Dim_ny_new)

    for i in range (t_dim):
        for j in range(x_dim):
            if np.absolute(temp_hist_l[i,j]- threshold) < tolerance:
                indices.append((i,j))
                if Dim_ny[i,j] < 3.0:
                    indices_nim.append((i,j)) # Indices of the Niyama number below threshold
    
    
    
    # print(indices_nim)          

    # print(Dim_ny)
    Niyama_pct = [Dim_ny[i,j] for i,j in indices]
    Niyama_array = np.array(Niyama_pct)
    # print(Niyama_array)
    Lowest_Niyama = round(np.min(Niyama_array),2)
    Avg_Niyama = np.mean(Niyama_array)
    # print(f"Lowest Niyama Number: {Lowest_Niyama}")

   
    if gen_graph:
        
        # Create a meshgrid for space and time coordinates
        space_coord, time_coord = np.meshgrid(np.arange(t_hist.shape[1]), np.arange(t_hist.shape[0]))
        space_coord = space_coord * dx
        time_coord = time_coord * dt 
        # Create a figure with two subplots
        fig, (ax1) = plt.subplots(1,figsize=(10, 6))

        # Plot the temperature history on the left subplot
        im1 = ax1.pcolormesh(space_coord, time_coord, t_hist, cmap='coolwarm', shading='auto')
        ax1.set_xlim(left=0, right=length,auto=True)
        ax1.set_ylim(0, current_time)
        
        ax1.set_xlabel('Space (mm)', fontname='Times New Roman', fontsize=16)
        
        ax1.set_ylabel('Time(Seconds)',fontname='Times New Roman', fontsize=16)
        ax1.set_title('Temperature Field',fontname='Times New Roman', fontsize=20)
        ax1.contour(space_coord, time_coord, t_hist, colors='red', linewidths=1.0, alpha=0.9)

        ax1.grid(True)
        cbar = fig.colorbar(im1, ax=ax1)
        cbar.ax.invert_yaxis()
        cbar.set_label('Temperature (K)', rotation=270, labelpad=20, fontname='Times New Roman', fontsize=16)
        
        # plt.contour(t_hist, colors='black', linewidths=0.5)
        
        
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
        plt.xlabel('Time(s)',fontname='Times New Roman', fontsize=16)
        plt.ylabel('Temperature (K)',fontname='Times New Roman', fontsize=16)
        plt.title('Cooling Curve @ 7.5',fontname='Times New Roman', fontsize=20) 
        plt.legend()
        plt.grid(True)
        plt.show()


        # Plot Niyama number distribution
       
        space_coord_1, time_coord_1 = np.meshgrid(np.arange(Dim_ny.shape[1]), np.arange(Dim_ny.shape[0]))
        # space_coord_1 = space_coord_1 * dx
        time_coord_1 = time_coord_1 * dt
        
        # norm = colors.Normalize(vmin= np.min(Dim_ny), vmax= np.max(Dim_ny), clip=False)
    
        if indices_nim:
            hlt_t, hlt_x = zip(*indices_nim) 
            real_t = [] # 
            for index in indices_nim:
                real_t.append(time_coord_1[index[0],index[1]])

        plt.figure(figsize=(10, 6))

        im1 =plt.pcolormesh(space_coord_1, time_coord_1, Dim_ny_new,cmap='viridis', shading='auto')
        if indices_nim:
            plt.scatter(hlt_x, real_t, color='red', s=20, marker='o', alpha=0.8,zorder=50, label='Heat Loss Threshold')
        # plt.set_xlim(left=0, right=length,auto=True)
        # plt.set_ylim(0, current_time)
        plt.xlabel('Space (mm)', fontname='Times New Roman', fontsize=16)
        plt.ylabel('Time',fontname='Times New Roman', fontsize=16)
        plt.xscale('linear')
        plt.yscale('linear')
        plt.rcParams['figure.dpi'] = 600
        plt.title('Evolution of Critical Niyama',fontname='Times New Roman', fontsize=20)
        # plt.contour(space_coord_1, time_coord_1, Dim_ny, colors='white', linewidths=1.0, alpha=0.9)
        plt.grid(True)
        cbar = plt.colorbar(im1)
        # cbar.ax.invert_yaxis()
        cbar.set_label('Niyama Number', rotation=270, labelpad=20, fontname='Times New Roman', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        dim_max = np.max(Dim_ny)
        print
        
        bins = [0,3, dim_max]
        cmap1 = plt.get_cmap('Reds')
        cmap2 = plt.get_cmap('Blues')

       
        masked_array1 = np.ma.masked_where((Dim_ny < bins[0]) | (Dim_ny > bins[1]), Dim_ny)
        masked_array2 = np.ma.masked_where((Dim_ny <= bins[1]) | (Dim_ny > bins[2]), Dim_ny)

        
        # Plot the first bin
        plt.pcolormesh(space_coord_1, time_coord_1, masked_array1, cmap=cmap1, shading='auto')

        # Plot the second bin
        plt.pcolormesh(space_coord_1, time_coord_1, masked_array2, cmap=cmap2, shading='auto')

        plt.xlabel('Space (mm)', fontname='Times New Roman', fontsize=16)
        plt.ylabel('Time', fontname='Times New Roman', fontsize=16)
        plt.title('Evolution of Critical Niyama', fontname='Times New Roman', fontsize=20)
        plt.grid(True)

        # Add colorbar with proper label
        cbar = plt.colorbar()
        cbar.set_label('Niyama Number', rotation=270, labelpad=20, fontname='Times New Roman', fontsize=16)
        cbar.set_ticks(np.linspace(0, dim_max, 20))
        plt.tight_layout()
        plt.show()
       

       # Create a new colormap that combines both 'Blues' and 'Oranges' for the full range from 0 to 10
        

        # # Combine the two colormaps
        # combined_cmap = ListedColormap(np.vstack((cmap1(np.linspace(0, 1, 128)), cmap2(np.linspace(0, 1, 128)))))

        # # Define the boundaries to match the bins and ensure proper scaling across the full data range
        # bounds = np.concatenate((np.linspace(0,3,128),np.linspace(3,dim_max,129)))  # Create more boundaries for smooth transitions
        # norm = BoundaryNorm(bounds, combined_cmap.N)

        # # Plotting using pcolormesh with the combined colormap
        # plt.figure(figsize=(10, 6))
        # plt.pcolormesh(space_coord_1, time_coord_1, Dim_ny, cmap=combined_cmap, norm=norm, shading='auto')

        # plt.xlabel('Space (mm)', fontname='Times New Roman', fontsize=16)
        # plt.ylabel('Time', fontname='Times New Roman', fontsize=16)
        # plt.title('Evolution of Critical Niyama', fontname='Times New Roman', fontsize=20)
        # plt.grid(True)

        # # Add colorbar with proper label
        # cbar = plt.colorbar()
        # cbar.set_label('Niyama Number', rotation=270, labelpad=20, fontname='Times New Roman', fontsize=16)
        # cbar.set_ticks(np.linspace(0, dim_max, 20))

        # plt.tight_layout()
        # plt.show()



    if gen_data:
        return t_hist
    
        

    




    # print(f'Lowest Niyama:{Lowest_Niyama}, rho_l:{rho_l}, rho_s:{rho_s}, k_l:{k_l}, k_s:{k_s}, cp_l:{cp_l}, cp_s:{cp_s}, t_surr:{t_surr}, L_fusion:{L_fusion}, temp_init:{temp_init},htc_l:{htc_l},htc_r:{htc_r},length:{length}')
    



