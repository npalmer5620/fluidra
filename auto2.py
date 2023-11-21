import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate

# Replace 'your_data.csv' with the path to your data file
data_file = 'data/48.csv'

# Read the data
data = pd.read_csv(data_file)

# Extract voltage values
voltage_values = data['CH1(V)']
time_values = data['Time(s)']

# Original sample rate calculation
original_sample_rate = 1 / ((time_values[len(time_values) - 1] - time_values[0]) / len(time_values))

# New sampling rate (at least 2x the highest frequency of interest)
new_sample_rate = 2500  # Hz

# 60 Hz bandstop filter (Butterworth)
bandstop_low = 59  # Hz
bandstop_high = 61  # Hz
bs_b, bs_a = butter(N=2, Wn=[bandstop_low, bandstop_high], fs=original_sample_rate, btype='bandstop')
bandstopped_signal = filtfilt(bs_b, bs_a, voltage_values)

# Design a low-pass filter (Butterworth)
cutoff_frequency = 1000  # Hz

b, a = butter(N=5, Wn=cutoff_frequency, fs=original_sample_rate, btype='lowpass')

# Apply the filter
filtered_signal = filtfilt(b, a, bandstopped_signal)

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)  # Two rows, one column, first plot
plt.plot(time_values, voltage_values)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')


# Downsample the signal
decimation_factor = int(original_sample_rate / new_sample_rate)
resampled_signal = filtered_signal[::decimation_factor]
resampled_time = time_values[::decimation_factor]

plt.subplot(3, 1, 2)  # Two rows, one column, second plot
plt.plot(resampled_time, resampled_signal)
plt.title('Resampled & Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')

# plt.show()

# Autocorrelation analysis
autocorr = correlate(resampled_signal, resampled_signal, mode='full', method='direct')
print(len(autocorr), autocorr)
# autocorr = autocorr[len(autocorr)//2:]  # Keep only the positive lags
print(len(autocorr), autocorr)

# Find the lag with the maximum correlation (peak)
peak_lag = np.argmax(autocorr)
print("Peak Lag =", autocorr[peak_lag], ":", peak_lag)

# Calculate the period
period = peak_lag / new_sample_rate

# Plotting
plt.subplot(3, 1, 3)  # Two rows, one column, second plot
plt.plot(autocorr)
plt.title('Autocorrelation of the Signal')
plt.xlabel('Lag (samples)')
plt.ylabel('Correlation coefficient')
plt.show()

print(f'Estimated period of the signal: {period} seconds')


exit()