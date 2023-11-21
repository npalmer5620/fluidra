import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import butter, filtfilt, correlate, hilbert, sosfilt, correlation_lags, decimate

MAX_SIGNAL_TIME = 1.0  # s
RESAMPLE_RATE = 2500.0


def new_proc(data, downsample_rate=RESAMPLE_RATE):
    # Extract voltage values
    voltage_signal = data['CH1(V)'].to_numpy(dtype=np.float64)
    time_axis = data['Time(s)'].to_numpy(dtype=np.float64)
    average_sampling_rate = 1.0 / np.mean(np.abs(np.diff(time_axis)))
    time_axis += 0.0 - time_axis[0]

    for i in range(len(time_axis)):
        if time_axis[i] >= MAX_SIGNAL_TIME:
            time_axis = time_axis[0:i]
            voltage_signal = voltage_signal[0:i]
            print("Signal truncated at", time_axis[len(time_axis) - 1], "s")
            break
    

    downsampled_signal = decimate(voltage_signal, int(average_sampling_rate / downsample_rate),
                                  ftype='fir', zero_phase=True)
    downsampled_time = np.linspace(time_axis[0], time_axis[len(time_axis) - 1] - time_axis[0], len(downsampled_signal), endpoint=False)

    return downsampled_signal, downsampled_time


def process(data, downsample_rate=RESAMPLE_RATE):
    # Extract voltage values
    voltage_signal = data['CH1(V)'].to_numpy(dtype=np.float64)
    time_axis = data['Time(s)'].to_numpy(dtype=np.float64)
    average_sampling_rate = 1.0 / np.mean(np.abs(np.diff(time_axis)))
    time_axis += 0.0 - time_axis[0]

    # Antialias the signal
    sos = butter(7, (downsample_rate / 2.0), fs=average_sampling_rate, btype='lowpass', analog=False, output='sos')
    filtered_signal = sosfilt(sos, voltage_signal)

    # Downsample the signal
    downsampling_factor = int(average_sampling_rate / downsample_rate)
    downsampled_signal = filtered_signal[::downsampling_factor]
    downsampled_time = time_axis[::downsampling_factor]

    return downsampled_signal, downsampled_time



# Read the data
# data_10 = pd.read_csv('data/10s2.csv', dtype=np.float64)
data_28 = pd.read_csv('data/50s2.csv', dtype=np.float64)
# data_48 = pd.read_csv('data/50s2.csv', dtype=np.float64)

s1, t1 = new_proc(data_28)
s2, t2 = new_proc(data_28)

# Plot the original signal
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t1, s1)
plt.title('Signal 1')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')

plt.subplot(3, 1, 2)
plt.plot(t2, s2)
plt.title('Signal 2')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')

# Autocorrelation analysis
corr = correlate(s1, s2, mode='full', method='auto')
lags = correlation_lags(len(s2), len(s1))[np.argmax(corr):] / RESAMPLE_RATE
corr = corr[np.argmax(corr):]
# corr = corr / np.max(corr)

# Find the lag with the maximum correlation (peak)
inflection = np.diff(np.sign(np.diff(corr))) # Find the second-order differences
peaks = (inflection < 0).nonzero()[0] + 1 # Find where they are negative

top5_indices = np.argpartition(corr[peaks], -5)[-5:]
print(corr[peaks])
print(top5_indices)

for i in top5_indices:
    print(corr[i])

#top5_indices = top5_indices[np.argsort(peaks[top5_indices])]
#top5_peaks = a[ind]

delay = peaks[corr[peaks].argmax()] # Of those, find the index with the maximum value
print("Delay =", delay, "samples")
signal_freq = RESAMPLE_RATE / delay

print("Signal Frequency =", signal_freq, "Hz")
print("Highest Lag at", lags[delay], f"({delay} samples)", "s", f"with coeff = {corr[delay]} (normalized = {corr[delay] / np.max(corr)})")

# Plotting
# lags = RESAMPLE_RATE / peaks
plt.subplot(3, 1, 3)
plt.plot(lags, corr / np.max(corr))

for index in peaks:
    plt.scatter(lags[index], corr[index] / np.max(corr), color='red')
    
plt.title('Correlation of the Signals')
plt.xlabel('Lag (s)')
plt.ylabel('Normalized Correlation')

plt.show()

