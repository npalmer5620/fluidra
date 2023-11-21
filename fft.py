import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Replace 'data/28.csv' with the path to your data file
data_file = 'data/10.csv'

# Read the data
data = pd.read_csv(data_file)

# Extract voltage values and time stamps
voltage_values = data['CH1(V)'].values
time_values = data['Time(s)'].values

# Calculate the sampling rate
sampling_rate = 1 / ((time_values[len(time_values) - 1] - time_values[0]) / len(time_values))

# Apply FFT on the voltage values
fft_result = np.fft.fft(voltage_values)
fft_freq = np.fft.fftfreq(len(voltage_values), d=(1 / sampling_rate))

# Only take the positive half of the spectrum, since it is symmetric around 0
half_n = len(fft_result) // 2
fft_result = fft_result[:half_n]
fft_freq = fft_freq[:half_n]

# Compute the magnitude of the FFT, which represents the energy
fft_magnitude = np.abs(fft_result)

# Plot the FFT result
plt.figure(figsize=(14, 7))
plt.plot(fft_freq, fft_magnitude)
plt.title('Frequency Spectrum of the Signal')
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 500)
plt.ylim(0, 1000)
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
