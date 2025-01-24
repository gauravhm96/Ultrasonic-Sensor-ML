import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class FftSignal:
    def __init__(self):
        self.folder_path = None

    def load_fft_data(self, file_path):
        signal_data = pd.read_csv(file_path, delimiter=None, header=None, sep='\s+', skiprows=1)
        return pd.DataFrame(signal_data)

    def plot_signals(self, df):
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        # Plot all the signals
        for i in range(df.shape[0]):
            # Ignoring the first 16 columns
            signal = df.iloc[i].iloc[16:]
            axs[0].plot(signal.values[:], label=f'Signal {i+1}')
        # Adding labels and title to the first subplot
        axs[0].grid(True)
        axs[0].set_xlabel('Time (samples)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Signal Data')
        
        # Plot all the signals with absolute values and mark peaks
        for i in range(df.shape[0]):
            # Ignoring the first 16 columns
            signal = df.iloc[i].iloc[16:]
            std_dev = np.std(np.abs(signal.values[:]))  # Use standard deviation as a measure of variation
            
            prominence = std_dev
            # Find peaks with a minimum prominence value (can be adjusted)
            peaks, _ = find_peaks(np.abs(signal.values[:]), prominence=prominence) 

            axs[1].plot(signal.values[:], label=f'Signal {i+1}')
            axs[1].plot(peaks, signal.values[peaks], 'rx', label=f'Peaks {i+1}')  # Marking the peaks in red

        # Adding labels and title to the second subplot
        axs[1].grid(True)
        axs[1].set_xlabel('Time (samples)')
        axs[1].set_ylabel('Absolute Amplitude')
        axs[1].set_title('Absolute Value of Signals With Peaks')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_40_Hand.txt'
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_70.txt'
    fft_signal = FftSignal()
    signal_data = fft_signal.load_fft_data(folder_path)

    fft_signal.plot_signals(signal_data)

    print("MATLAB-style FFT plots saved successfully.")
