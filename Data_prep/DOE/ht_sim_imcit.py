import numpy as np
import matplotlib.pyplot as plt

def sim1d_im(rho_l, rho_s, k_l, k_s, cp_l, cp_s, bc_l, bc_r, L_fusion, temp_init):
    

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
    time_end = 20                 # seconds                         
    #num_steps = 10000
    # dt = time_end/num_steps
    #dt = abs(0.5 *(dx**2/(max(alpha_l, alpha_s, alpha_m)))) # Time step size
    #dt = abs(dx**2/(2*(max(alpha_l,alpha_m,alpha_s)))) # Time step size
    num_steps = 1000  # Number of time steps
    dt = time_end / num_steps
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
    

    # Coefficients for the implicit method
    beta_l = alpha_l * dt / (2 * dx**2)
    beta_s = alpha_s * dt / (2 * dx**2)
    beta_m = alpha_m * dt / (2 * dx**2)


    def thomas_algorithm(a, b, c, d):
        n = len(d)
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
    
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
    
        for i in range(1, n-1):
            c_prime[i] = c[i] / (b[i] - a[i-1] * c_prime[i-1])
    
        for i in range(1, n):
            d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / (b[i] - a[i-1] * c_prime[i-1])
    
        x = np.zeros(n)
        x[-1] = d_prime[-1]
    
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
        return x
    
    
    # Time-stepping loop
    for m in range(1, num_steps + 1):
        # Construct the matrix and RHS
        a = np.zeros(num_points - 2)
        b = np.zeros(num_points - 1)
        c = np.zeros(num_points - 2)
        d = np.zeros(num_points - 1)
    
        for n in range(1, num_points - 1):
            if temperature[n] >= T_L:
                beta = beta_l
            elif T_S < temperature[n] < T_L:
                beta = beta_m
            else:
                beta = beta_s
        
            a[n-1] = -beta
            b[n-1] = 1 + 2 * beta
            c[n-1] = -beta
            d[n-1] = (1 - 2 * beta) * temperature[n] + beta * (temperature[n+1] + temperature[n-1])
        
            if T_S < temperature[n] < T_L:
                phase[n] = (T_L - temperature[n]) / (T_L - T_S)
            elif temperature[n] < T_S:
                phase[n] = 1
            else:
                phase[n] = 0
    
        # Solve the tridiagonal system
        temperature[1:num_points] = thomas_algorithm(a, b, c, d)
    
        temperature_history.append(temperature.copy())
        phi_history.append(phase.copy())

    # Compute the average phase at the end of the simulation
    final_phase = phi_history[-1]
    
    average_solid_fraction = np.mean(final_phase)

    #Assuming you have temperature_history and phi_history as lists of arrays
    temperature_history = np.array(temperature_history)
    phi_history = np.array(phi_history)



    # Check the new shape after transposing
    #print("Transposed Temperature History Shape:", temperature_history.shape)
    #print("Transposed Phi History Shape:", phi_history.shape)

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

    #plot the main
    #fig, ax = plt.subplots(figsize=(14, 6))
    #im = ax.pcolormesh(space_coord, time_coord, Niyama, cmap='viridis')
    #ax.set_xlabel('Space Coordinate')
    #ax.set_ylabel('Time')
    #ax.set_title('Main Variation Over Time')
    #fig.colorbar(im, ax=ax, label='Main')
    #plt.tight_layout()
    #plt.show()
    
    

    
    return average_solid_fraction
