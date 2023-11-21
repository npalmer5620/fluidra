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

sample_time = (time_values[len(time_values) - 1] - time_values[0])
sample_dt = sample_time / len(time_values)
sample_freq = 1 / sample_dt

resample_rate = 5000  # Hz
samples_per_resample = int(sample_freq / resample_rate)
new_sample_freq = sample_freq / samples_per_resample
resample_dt = 1 / new_sample_freq

print(f"Sample Time: {sample_time} s")
print(f"Sample dt: {sample_dt} s")
print(f"Sample Frequency: {sample_freq} Hz")
print(f"Resample Rate: {resample_rate} Hz")
print(f"Samples per Resample: {samples_per_resample}")
print(f"New Sample Frequency: {new_sample_freq} Hz")

resampled_voltages = []
# temp_mean = 0.0

# Resample the data
for i in range(len(voltage_values)):
    if (i % samples_per_resample == 0):
        resampled_voltages.append(voltage_values[i])

print(f"N Resampled Voltages: {len(resampled_voltages)}")
print(f"Resample dt: {resample_dt} s")

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)  # Two rows, one column, first plot
plt.plot(time_values, voltage_values)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')

# Plot the resampled signal
# For the resampled signal, you need to create a new time axis
resampled_time_values = np.arange(len(resampled_voltages)) * resample_dt

plt.subplot(2, 1, 2)  # Two rows, one column, second plot
plt.plot(resampled_time_values, resampled_voltages)
plt.title('Resampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')

plt.show()

exit()

# Define the range of time delays to test
max_time_delay = len(resampled_voltages)  # For example, test delays up to 50 samples

# Initialize variables to store the best correlation and corresponding delay
best_corr = 0
best_delay = 0

# Iterate over the range of time delays
for time_delay in range(max_time_delay + 1):
    print(f"{time_delay} of {max_time_delay}", end="\t")

    voltage_delayed = np.roll(resampled_voltages, time_delay)
    voltage_delayed[:time_delay] = 0  # Zeroing the shifted part

    # Compute cross-correlation
    cross_corr = np.correlate(resampled_voltages, voltage_delayed, mode='full')
    max_corr = max(cross_corr)  # Get the maximum correlation value for this delay

    print(f"Max Corr = {max_corr}")

    # Check if this is the best correlation so far
    if max_corr > best_corr:
        best_corr = max_corr
        best_delay = time_delay

# Output the best time delay
print(f"Best time delay: {best_delay} samples")

# Plot the cross-correlation for the best time delay
voltage_delayed = np.roll(resampled_voltages, best_delay)
voltage_delayed[:best_delay] = 0
cross_corr = np.correlate(resampled_voltages, voltage_delayed, mode='full')
cross_corr = cross_corr[cross_corr.size // 2:]

plt.plot(cross_corr)
plt.title(f'Cross-Correlation with Best Delay ({best_delay} samples)')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.show()
