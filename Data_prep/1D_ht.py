import numpy as np # import numpy library as np
import matplotlib.pyplot as plt  # import matplotlib.pyplot library as plt
import csv # import csv module

# Parameters
alpha = 0.01  # Thermal diffusivity
length = 1.0  # Length of the rod
time_end = 1.0  # End time
num_points = 100  # Number of spatial points
num_steps = 1000  # Number of time steps

# Spatial and time step sizes
dx = length / (num_points - 1) # Grid spacing
dt = time_end / num_steps # Time step size

# Initial condition
initial_temperature = np.zeros(num_points) # Initialize temperature array with zeros

# Set one end at 25 degrees and the other end at 600 degrees
initial_temperature[0] = 100.0  # Set the first element (one end) to 25 degrees
initial_temperature[-1] = 600.0  # Set the last element (other end) to 600 degrees

# Initialize temperature array
temperature = initial_temperature.copy() # Copy initial_temperature to temperature

# Finite difference method
for n in range(1, num_steps + 1):# Loop through time steps
    # Compute new temperature values using finite difference
    temperature[1:-1] = (
        temperature[1:-1]
        + alpha * dt / dx**2 * (temperature[2:] - 2 * temperature[1:-1] + temperature[:-2])
    )

# Plot the results
plt.plot(np.linspace(0, length, num_points), temperature)
plt.xlabel('Spatial Coordinate (x)')
plt.ylabel('Temperature (T)')
plt.title('1D Heat Transfer')
plt.show()

# Save data to CSV file
data = np.column_stack((np.linspace(0, length, num_points), temperature))
csv_filename = 'heat_transfer_data.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Spatial Coordinate (x)', 'Temperature (T)'])
    csv_writer.writerows(data)

print(f'Data saved to {csv_filename}')

