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
            
    def plot_signal_with_peaks(self,dataframe, peak_index, peak_value):
        plt.figure(figsize=(15, 8))
        for index, row in dataframe.iterrows():
            plt.plot(row, color='lightgray', linewidth=0.5, alpha=0.7)  # Plotting each signal

        # Highlight the most prominent peak on all signals using lines
        plt.axvline(x=peak_index, color='red', linestyle='--', label='Peak Index')
        plt.axhline(y=peak_value, color='blue', linestyle='--', label='Peak Amplitude')

        plt.title('All Filtered ADC Signals with Highlighted Peak')
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
    
    def detect_prominent_peaks(self,dataframe,std_multiplier=3):
        max_signal = dataframe.max(axis=0).to_numpy()
        
        # Set prominence threshold dynamically
        prominence_threshold = np.mean(max_signal) + std_multiplier * (np.std(max_signal))

        # Find peaks in the mean signal with the specified prominence
        peaks, properties = find_peaks(max_signal, prominence=prominence_threshold)

        if len(peaks) > 0:
           prominent_peak_index = peaks[0]  # Take the first prominent peak
           prominent_peak_value = max_signal[prominent_peak_index]
           return prominent_peak_index, prominent_peak_value
        return None, None
    
    def calculate_distance_from_peak(self,highest_peak_index):
        ADC_MAX_SAMPLE_FREQUENCY = 125000000  # Hz
        # # Convert TOF to distance
        SPEED_OF_SOUND = 343  # Speed of sound in meters per second
        MICROSECONDS_TO_SECONDS = 1e-6  # Conversion factor from microseconds to seconds

        # Convert peak index to time of flight in microseconds
        time_of_flight_us = highest_peak_index * (1 / ADC_MAX_SAMPLE_FREQUENCY) * 1e6

        # Convert time of flight to seconds
        time_of_flight_seconds = time_of_flight_us * MICROSECONDS_TO_SECONDS

        # Calculate distance in meters
        distance_meters = (time_of_flight_seconds * SPEED_OF_SOUND) / 2

        # Convert distance to centimeters
        distance_cm = distance_meters * 100

        return distance_cm
    
class TrainADC:
    def __init__(self):
        pass
    
    
    
if __name__ == "__main__":
    # folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_Me.txt'

    FILE_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Raw_Data/adc_67.txt"
    OUTPUT_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Test/adc_120.txt"

    myadcdata = ADCSignal()
    processadcdata = ADCSignalProcess()
    myplot = Plot_signals()

    my_adc_data = myadcdata.get_adc_data(FILE_PATH)
    myplot.plot_adc_data(my_adc_data)

    adc_data_array = my_adc_data.to_numpy()
    lowcut = 39500.0  # Lower cutoff frequency in Hz
    highcut = 41500.0  # Upper cutoff frequency in Hz
    fs = 1.953125e6    # Sampling frequency in Hz (as per your ADC configuration)
    
    sos = processadcdata.butter_bandpass(lowcut,highcut,fs)
    filtered_data_array = np.apply_along_axis(processadcdata.apply_bandpass_filter, 1, adc_data_array, sos)
    filtered_myadc_data = pd.DataFrame(filtered_data_array)
    #myplot.plot_adc_data(filtered_myadc_data)
    
    peak_index, peak_value = processadcdata.detect_prominent_peaks(filtered_myadc_data)
    print(processadcdata.calculate_distance_from_peak(peak_index))
    myplot.plot_signal_with_peaks(filtered_myadc_data,peak_index, peak_value)

    