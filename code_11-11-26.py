import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, butter, sosfilt
 
# Parameters
duration = 10
fs = 48000
f0 = 18000
f1 = 22000
 
# Time vector
N = int(duration * fs)
t = np.arange(N) / fs
 
# Generate linear CHIRP
x_lin = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
 
# Play / Record audio (mono)
print("Recording...")
 
myrecording = sd.playrec(
    x_lin,
    samplerate=fs,
    channels=1,
    dtype='float32'
)
 
sd.wait()
 
# High-pass filter at 17500 Hz
sos = butter(4, 17500, btype='highpass', fs=fs, output='sos')
filtered_signal = sosfilt(sos, myrecording[:, 0])
 
# Time vector
t_record = np.arange(len(myrecording)) / fs
 
print("Recording finished!")
 
# FFT of original signal
fft_original = np.fft.fftshift(np.fft.fft(myrecording[:, 0]))
freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
 
# FFT of filtered signal
fft_filtered = np.fft.fftshift(np.fft.fft(filtered_signal))
 
# Plot signals and FFT
fig, ax = plt.subplots(4, 1, figsize=(12, 12))
 
# Original recording
ax[0].plot(t_record, myrecording[:, 0])
ax[0].set_title("Original Recorded Signal")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amplitude")
 
# Filtered recording
ax[1].plot(t_record, filtered_signal)
ax[1].set_title("High-pass Filtered Signal (4000 Hz)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")
 
# FFT original
ax[2].plot(freq, np.abs(fft_original))
ax[2].set_title("FFT - Original Signal")
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Magnitude")
ax[2].set_xlim(0, fs/2)
 
# FFT filtered
ax[3].plot(freq, np.abs(fft_filtered))
ax[3].set_title("FFT - Filtered Signal")
ax[3].set_xlabel("Frequency (Hz)")
ax[3].set_ylabel("Magnitude")
ax[3].set_xlim(0, fs/2)
 
plt.tight_layout()
plt.show()