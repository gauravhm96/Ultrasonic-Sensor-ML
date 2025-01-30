import numpy as np
import pandas as pd

# Example raw amplitude data (200 time samples, 5 frequency bins)
raw_data = np.random.random((200, 5)) * 100  # Random amplitude data between 0 and 100
amplitude_buffer = pd.DataFrame(raw_data)

print(amplitude_buffer.head())


# Calculate row-wise mean and max amplitude
mean_amplitude = amplitude_buffer.mean(axis=1)  # Mean along each row (axis=1)
max_amplitude = amplitude_buffer.max(axis=1)    # Max along each row (axis=1)

# Create a new DataFrame to store the extracted features
features_df = pd.DataFrame({
    'Mean Amplitude': mean_amplitude,
    'Max Amplitude': max_amplitude
})

print(features_df.head())

import matplotlib.pyplot as plt

# Assuming raw_signal is the original amplitude buffer (200x85)
# Assuming features_df contains the extracted features (Mean Amplitude and Max Amplitude)

plt.figure(figsize=(14, 8))

# Plot Raw Signal (First column of amplitude buffer)
plt.subplot(3, 1, 1)
plt.plot(amplitude_buffer.iloc[0, :], label='Raw Signal (First Time Sample)', color='gray')
plt.title('Raw Signal (First Time Sample) vs Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Plot Mean Amplitude
plt.subplot(3, 1, 2)
plt.plot(features_df['Mean Amplitude'], label='Mean Amplitude', color='blue')
plt.title('Mean Amplitude vs Time')
plt.xlabel('Time (Samples)')
plt.ylabel('Mean Amplitude')
plt.grid(True)
plt.legend()

# Plot Max Amplitude
plt.subplot(3, 1, 3)
plt.plot(features_df['Max Amplitude'], label='Max Amplitude', color='red')
plt.title('Max Amplitude vs Time')
plt.xlabel('Time (Samples)')
plt.ylabel('Max Amplitude')
plt.grid(True)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


