import numpy as np
import matplotlib.pyplot as plt

# Create a Hanning window of length 100
N = 100
hanning_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

# Plot the window
plt.plot(hanning_window)
plt.title("Hanning Window")
plt.xlabel("Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
