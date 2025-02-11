import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks,butter, sosfiltfilt,welch
from scipy.ndimage import gaussian_filter1d


class ADCSignal:
    def __init__(self):
        self.file_path = None

    def get_adc_data(self, file_path):
        if not isinstance(file_path, str):
            print("Invalid file path provided.")
            return None
        self.file_path = file_path
        try:
            raw_adc_signal_data = pd.read_csv(file_path, delimiter="\t", header=None)
            df = pd.DataFrame(raw_adc_signal_data)
            removeheader_adc_data = df.iloc[:, 16:]

            myadc_data = removeheader_adc_data
            print("Signal data successfully fetched.")
            return myadc_data

        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except pd.errors.EmptyDataError:
            print("Error: File is empty.")
        except pd.errors.ParserError:
            print("Error: Could not parse the file correctly.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save_adc_data(self, adc_data, output_path):
        if adc_data is None or adc_data.empty:
            print("No processed ADC data to save.")
            return
        try:
            adc_data.to_csv(output_path, index=False, header=False, sep="\t")
            print(f"Processed data successfully saved to {output_path}")
        except Exception as e:
            print(f"An error occurred while saving the data: {e}")

class Plot_signals:
    def __init__(self):
        pass
    
    def plot_adc_data(self, adc_data):
        if adc_data is None or adc_data.empty:
            print("No processed ADC data to plot.")
            return

        try:
            # Assuming the adc_data has multiple columns, plot the first column against its index
            plt.figure(figsize=(10, 6))
            for i in range(adc_data.shape[0]):
                plt.plot(adc_data.columns, adc_data.iloc[i, :], label=f"Row {i + 1}")

            plt.title("ADC Signal Data")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            print("Plot displayed successfully.")
        except Exception as e:
            print(f"An error occurred while plotting the data: {e}")
            
    def plot_peaks_on_signal(self,dataframe, peaks_indices):
        num_signals = dataframe.shape[0]

        for i in range(num_signals):
            signal = dataframe.iloc[i, :].to_numpy()
            peaks = peaks_indices[i]

            plt.figure(figsize=(10, 6))
            plt.plot(signal, label=f'Signal {i+1}')
            plt.scatter(peaks, signal[peaks], color='red', label='Detected Peaks')
            plt.title(f'Signal {i+1} with Detected Peaks')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
class ADCSignalProcess:
    def __init__(self):
        pass

    # Designs a Butterworth band-pass filter.
    # Parameters:
    # - lowcut: Lower cutoff frequency in Hz.
    # - highcut: Upper cutoff frequency in Hz.
    # - fs: Sampling frequency in Hz.
    # - order: Order of the filter (default is 5).
    # Returns:
    # - sos: Second-order sections for the filter.
    def butter_bandpass(self,lowcut, highcut, fs):
        order=5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos
    
    # Applies a Butterworth band-pass filter to the data.
    # Parameters:
    # - data: Input signal data (1D NumPy array).
    # - lowcut: Lower cutoff frequency in Hz.
    # - highcut: Upper cutoff frequency in Hz.
    # - fs: Sampling frequency in Hz.
    # - order: Order of the filter (default is 5).
    # Returns:
    # - Filtered data as a NumPy array.
    def apply_bandpass_filter(self, data,sos):
        filtered_data = sosfiltfilt(sos, data, axis=-1)
        return filtered_data
    
    def detect_prominent_peaks(self,dataframe, std_multiplier=3):

        data_array = dataframe.to_numpy()
        absolute_signal = np.abs(data_array)
        mean_signal = np.mean(absolute_signal, axis=0)
        std_signal = np.std(absolute_signal, axis=0)
        
        prominence_threshold = mean_signal + std_multiplier * std_signal
        
        peaks_indices = []

        for col_idx in range(data_array.shape[1]):
            # Find peaks in the current column with the specified prominence
            peaks, _ = find_peaks(absolute_signal[:, col_idx], prominence=prominence_threshold[col_idx])
            peaks_indices.append(peaks)
        
        return peaks_indices
    
if __name__ == "__main__":
    # folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_Me.txt'

    FILE_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Raw_Data/adc_70.txt"
    OUTPUT_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Test/adc_120.txt"

    myadcdata = ADCSignal()
    processadcdata = ADCSignalProcess()
    myplot = Plot_signals()

    my_adc_data = myadcdata.get_adc_data(FILE_PATH)

    lowcut = 39000.0  # Lower cutoff frequency in Hz
    highcut = 41500.0  # Upper cutoff frequency in Hz
    fs = 1.953125e6    # Sampling frequency in Hz (as per your ADC configuration)
    
    sos = processadcdata.butter_bandpass(lowcut,highcut,fs)
    
    
    adc_data_array = my_adc_data.to_numpy()
    
    filtered_data_array = np.apply_along_axis(processadcdata.apply_bandpass_filter, 1, adc_data_array, sos)
    filtered_myadc_data = pd.DataFrame(filtered_data_array)
    print(filtered_myadc_data.shape)
    
    peaks = processadcdata.detect_prominent_peaks(filtered_myadc_data, std_multiplier=3)
    print(len(peaks))
    myplot.plot_peaks_on_signal(filtered_myadc_data,peaks)