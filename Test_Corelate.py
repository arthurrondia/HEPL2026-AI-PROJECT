import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

# Parameters
fs = 48000
f0 = 18000
f1 = 22000
frequency_step = 200
step_duration = 0.02
number_of_sweeps = 10

frequencies = np.arange(f0, f1 + frequency_step, frequency_step)
samples_per_step = int(step_duration * fs)

signal = []
phase = 0

# Generate signal
for i in range(number_of_sweeps):

    for f in frequencies:

        t_step = np.arange(samples_per_step) / fs

        x_step = np.sin(2 * np.pi * f * t_step + phase)

        phase = (phase + 2 * np.pi * f * step_duration) % (2 * np.pi)

        signal.extend(x_step)

x_signal = np.array(signal, dtype=np.float32)

# Add silence after the signal
silence = np.zeros(int(0.1 * fs), dtype=np.float32)
x_play = np.concatenate((x_signal, silence))

# Play and record
print("Playing and recording...")

myrecording = sd.playrec(
    x_play,
    samplerate=fs,
    channels=1,
    dtype='float32'
)

sd.wait()

received = myrecording[:, 0]

# Plot received signal spectrogram
plt.figure(figsize=(12, 6))

plt.specgram(
    received,
    NFFT=1024,
    Fs=fs,
    noverlap=768,
    cmap='viridis'
)

plt.ylim(15000, 25000)
plt.colorbar(label="Amplitude")
plt.title("Received Signal Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

# Cross-correlation
correlation = correlate(received, x_signal, mode='full', method='fft')
lags = correlation_lags(len(received), len(x_signal), mode='full')

# Keep positive delays
mask = lags >= 0

lags = lags[mask]
correlation = correlation[mask]

# Plot correlation
plt.figure(figsize=(12, 6))

plt.plot(lags, np.abs(correlation))
plt.xlim(0, 25000)

plt.xlabel("Delay (samples)")
plt.ylabel("Correlation Amplitude")
plt.title("Cross-Correlation")
plt.grid()

plt.tight_layout()
plt.show()

print("Finished!")