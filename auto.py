import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_data.csv' with the path to your data file
data_file = 'data/28.csv'

# Read the data
data = pd.read_csv(data_file)

# Extract voltage values
voltage_values = data['CH1(V)']
time_values = data['Time(s)']

resample_rate = 5000  # Hz
sample_freq = len(voltage_values) / time_values[len(time_values) - 1]
samples_per_resample = int(len(voltage_values) / (time_values[len(time_values) - 1] * resample_rate))
new_sample_freq = sample_freq / samples_per_resample

print(f"Sample Frequency: {sample_freq} Hz")
print(f"Resample Rate: {resample_rate} Hz")
print(f"Samples per Resample: {samples_per_resample}")
print(f"New Sample Frequency: {new_sample_freq} Hz")

resampled_voltage = []
temp_mean = 0.0

# Resample the data
for i in range(len(voltage_values)):
    if (i % samples_per_resample == 0) and (i != 0):
        temp_mean /= samples_per_resample
        resampled_voltage.append(temp_mean)
        # print(f"Sampled @ {i} of {len(voltage_values)}\t{temp_mean} V")
        temp_mean = 0.0
    else:
        temp_mean += voltage_values[i]

print(f"N Resampled Voltages: {len(resampled_voltage)}")
dt = time_values[samples_per_resample-1] - time_values[0]
print(f"dt: {dt} s")
print(f"ts: {dt * len(resampled_voltage)}")

exit()

# Define the range of time delays to test
max_time_delay = 50  # For example, test delays up to 50 samples

# Initialize variables to store the best correlation and corresponding delay
best_corr = 0
best_delay = 0

# Iterate over the range of time delays
for time_delay in range(max_time_delay + 1):
    print(f"{time_delay} of {max_time_delay}", end="\t")

    voltage_delayed = np.roll(voltage_values, time_delay)
    voltage_delayed[:time_delay] = 0  # Zeroing the shifted part

    # Compute cross-correlation
    cross_corr = np.correlate(voltage_values, voltage_delayed, mode='full')
    max_corr = max(cross_corr)  # Get the maximum correlation value for this delay

    print(f"Max Corr = {max_corr}")

    # Check if this is the best correlation so far
    if max_corr > best_corr:
        best_corr = max_corr
        best_delay = time_delay

# Output the best time delay
print(f"Best time delay: {best_delay} samples")

# Plot the cross-correlation for the best time delay
voltage_delayed = np.roll(voltage_values, best_delay)
voltage_delayed[:best_delay] = 0
cross_corr = np.correlate(voltage_values, voltage_delayed, mode='full')
cross_corr = cross_corr[cross_corr.size // 2:]

plt.plot(cross_corr)
plt.title(f'Cross-Correlation with Best Delay ({best_delay} samples)')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.show()
