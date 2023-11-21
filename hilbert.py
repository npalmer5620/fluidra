import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import butter, filtfilt, correlate, hilbert, sosfilt

# Replace 'your_data.csv' with the path to your data file
data_file = 'data/28.csv'

# Read the data
data = pd.read_csv(data_file, dtype=np.float64)

# Extract voltage values
voltage_values = data['CH1(V)'].to_numpy(dtype=np.float64)
time_values = data['Time(s)'].to_numpy(dtype=np.float64)
time_values += 0 - time_values[0]

# Calculate the average sampling rate
average_sampling_rate = 1 / data['Time(s)'].diff().abs().mean()

# Antialiasing Filter the signal
downsample_rate = 2500  # Hz
sos = butter(7, [1, (downsample_rate / 2.0)], fs=average_sampling_rate, btype='bandpass', analog=False, output='sos')
filtered_signal = sosfilt(sos, voltage_values)

# Downsample the signal
downsampling_factor = int(average_sampling_rate / downsample_rate)
downsampled_signal = filtered_signal[::downsampling_factor]
downsampled_time = time_values[::downsampling_factor]

# Perform Empirical Mode Decomposition (EMD)
emd = EMD()
IMFs = emd(downsampled_signal)

# Analyze each IMF
for i, imf in enumerate(IMFs, start=1):
    # Apply the Hilbert Transform
    analytic_signal = hilbert(imf)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * downsample_rate

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(downsampled_time[:-1], instantaneous_frequency)
    plt.title(f'Instantaneous Frequency of IMF {i}')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.subplot(2, 1, 2)
    plt.plot(downsampled_time, imf)
    plt.title(f'IMF {i}')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    # Check if IMF falls in the 1-1000 Hz range
    if any((1 <= freq <= 1000) for freq in instantaneous_frequency):
        print(f"IMF {i}\tmean f = {np.mean(instantaneous_frequency)} f\tstd dev f = {np.std(instantaneous_frequency)} f")
