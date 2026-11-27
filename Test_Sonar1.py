import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Parameters
fs = 48000
f0 = 18000
f1 = 22000
frequency_step = 200
step_duration = 0.01
number_of_sweeps = 5

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

print("Playing and recording...")

myrecording = sd.playrec(
    x_signal,
    samplerate=fs,
    channels=1,
    dtype='float32'
)

sd.wait()

# Plot spectrogram
plt.figure(figsize=(12, 6))

plt.specgram(
    myrecording[:, 0],
    NFFT=1024,
    Fs=fs,
    noverlap=768,
    cmap='viridis'
)

plt.ylim(15000, 25000)
plt.colorbar(label="Amplitude (dB)")
plt.title("Received Signal Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()