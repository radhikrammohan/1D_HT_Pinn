import numpy as np
import matplotlib.pyplot as plt

def sim1d(rho_l, rho_s, k_l, k_s, cp_l, cp_s, bc_l, bc_r, L_fusion, temp_init):
    

    # Geometry
    length = 1e-2                # Length of the rod

    T_L = 889.0                  # Liquidus temperature
    T_S = T_L-338.3                    # Solidus temperature
    
    rho_m = (rho_l +rho_s) /2 # Average density
    k_m = (k_l + k_s) / 2 # Average thermal conductivity    
    cp_m = (cp_l + cp_s) / 2 # Average heat capacity

    alpha_l = k_l / (rho_l * cp_l)  # Thermal diffusivity of the liquid
    alpha_s = k_s / (rho_s * cp_s)  # Thermal diffusivity of the solid
    alpha_m = k_m / (rho_m * cp_m)  # Thermal diffusivity of the solid
    
    # Spatial discretization

    num_points = 50                  # Number of spatial points
    dx = length / (num_points - 1)   # Grid spacing
    
                                                                 
    # Time Discretization  
    time_end = 30                # seconds                         
    #num_steps = 10000
    # dt = time_end/num_steps
    dt = abs(0.5 *(dx**2/(max(alpha_l, alpha_s, alpha_m)))) # Time step size
    #dt = abs(dx**2/(2*(max(alpha_l,alpha_s)))) # Time step size
    num_steps = round(time_end/dt) +1    # Number of time steps
    print('Number of time steps:', num_steps)
    
    
   
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

    #Assuming you have temperature_history and phi_history as lists of arrays
    temperature_history = np.array(temperature_history)
    phi_history = np.array(phi_history)



    # Check the new shape after transposing
    print("Transposed Temperature History Shape:", temperature_history.shape)
    print("Transposed Phi History Shape:", phi_history.shape)

    # Create a meshgrid for space and time coordinates
    space_coord, time_coord = np.meshgrid(np.arange(temperature_history.shape[1]), np.arange(temperature_history.shape[0]))

    time_coord = time_coord * dt
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the temperature history on the left subplot
    im1 = ax1.pcolormesh(space_coord, time_coord, temperature_history, cmap='viridis')
    ax1.set_xlabel('Space Coordinate')
    ax1.set_ylabel('Time')
    ax1.set_title('Temperature Variation Over Time')
    fig.colorbar(im1, ax=ax1, label='Temperature')

    # Plot the phase history on the right subplot
    im2 = ax2.pcolormesh(space_coord, time_coord, phi_history, cmap='viridis')
    ax2.set_xlabel('Space Coordinate')
    ax2.set_ylabel('Time')
    ax2.set_title('Phase Variation Over Time')
    fig.colorbar(im2, ax=ax2, label='Phase')
    plt.tight_layout()
    plt.show()

    # plot the main
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(space_coord, time_coord, Niyama, cmap='viridis')
    ax.set_xlabel('Space Coordinate')
    ax.set_ylabel('Time')
    ax.set_title('Main Variation Over Time')
    fig.colorbar(im, ax=ax, label='Main')
    plt.tight_layout()
    plt.show()
    
    

    
    return average_solid_fraction
