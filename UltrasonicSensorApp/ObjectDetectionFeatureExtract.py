# signal_processor.py

import pandas as pd
import numpy as np
from SignalProcess import SignalProcessor
import os
import datetime
import matplotlib.pyplot as plt
import random

class FeatureExtract:
    def __init__(self):
        pass

    
    def apply_threshold_filtering(self,updated_signal_dictionary):
    
        if not updated_signal_dictionary:
            raise ValueError("updated_signal_dictionary is empty. Perform NoiseFiltering or related preprocessing first.")

        # Extract all threshold values from updated_signal_dictionary
        threshold_values = [element[5] for element in updated_signal_dictionary]

        # Calculate the overall threshold as the median of all threshold values
        overall_threshold = np.median(threshold_values)

        # Create a new list to hold the updated signals
        updated_signals = []

        # Iterate over the updated_signal_dictionary and filter the signals
        for key, signal, peaks, median_peak, median_std, threshold, filtered_peaks in updated_signal_dictionary:
            
            filtered_signal = signal.copy()
            filtered_signal[filtered_signal > overall_threshold] = 0
            updated_signals.append((key, filtered_signal, peaks, median_peak, median_std, threshold, filtered_peaks))

        # Find the time index corresponding to the overall threshold (for the last processed signal)
        # Note: Ensure `signal` refers to the correct variable for this calculation
        overall_threshold_index = np.argmax(signal.values > overall_threshold)

        return updated_signals, overall_threshold
    
    def extract_peak_windows(self, updated_signal_dictionary):
        # Initializing the list to store the selected windows with peaks
        selected_peak_windows = []

        # Looping through updated_signal_dictionary to get the signal data and real peak values
        for i, element in enumerate(updated_signal_dictionary):
            # Access key, filtered_signal, peaks, median_peak, median_std, threshold, and filtered_peaks
            key = element[0]  # Assuming key is the first element
            filtered_signal = element[1]  # Assuming filtered_signal is the second element (Pandas Series)
            peaks = element[2]  # Assuming peaks are the third element
            median_peak = element[3]  # Median peak value
            median_std = element[4]  # Median standard deviation
            threshold = element[5]  # Threshold value
            filtered_peaks = element[6]  # Peaks filtered by threshold

            # Determine the window size dynamically based on the distances between consecutive peaks,
            # median peak, median std, and threshold
            if len(peaks) > 1:
                window_size = np.min(np.diff(peaks))  # Use the minimum distance between consecutive peaks
            else:
                window_size = len(filtered_signal)  # Use the entire length of the signal if only one peak is detected

            # Optionally, you can adjust the window size based on median peak, median std, and threshold
            window_size *= 2  # Adjust the multiplier as needed

            # Loop through signal in steps of window_size
            for k in range(0, len(filtered_signal), window_size):
                window = filtered_signal.iloc[k:k + window_size]  # Use iloc for integer-based indexing
                if median_peak in window.values:  # Check if median_peak in window values
                    selected_peak_windows.append(window)  # Append window to selected_peak_windows

        # Return the selected peak windows
        return selected_peak_windows
    
    def calulate_window(self,num_samples_per_window):
        # Constants
        ADC_MAX_SAMPLE_FREQUENCY = 125000000  # Hz
        ADC_SAMPLE_DECIMATION = 64
        ADC_SAMPLE_FREQUENCY = ADC_MAX_SAMPLE_FREQUENCY / ADC_SAMPLE_DECIMATION  # Hz

        # Calculate the duration of each window
        duration_per_sample = 1 / ADC_SAMPLE_FREQUENCY  # Duration of each sample in seconds
        duration_per_window = num_samples_per_window * duration_per_sample  # Duration of each window in seconds
        
        return duration_per_window, ADC_SAMPLE_FREQUENCY
    
    def save_PeakSspectrogramsType_1(self, selected_peak_windows, updated_signal_dictionary, folder_path, num_samples_per_window, ADC_SAMPLE_FREQUENCY):
        """
        Generate and save spectrograms for selected windows.
        
        Args:
            selected_peak_windows (list): List of windows containing the selected peaks.
            updated_signal_dictionary (list): Dictionary of updated signals with metadata.
            folder_path (str): Path to save the spectrogram images.
            num_samples_per_window (int): Number of samples per window.
            ADC_SAMPLE_FREQUENCY (float): Sampling frequency of the ADC.
        """
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Calculate the duration per window
        duration_per_sample = 1 / ADC_SAMPLE_FREQUENCY  # Duration of each sample in seconds
        duration_per_window = num_samples_per_window * duration_per_sample  # Duration of each window in seconds

        # Loop through the selected peak windows
        for i, window in enumerate(selected_peak_windows):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.specgram(window, Fs=ADC_SAMPLE_FREQUENCY, mode='psd', scale='dB', cmap='gray')

            # Set labels and title
            ax.set_xlabel('Time (seconds)')  # Update the x-axis label to show time in seconds
            ax.set_ylabel('Frequency')
            ax.set_title('Spectrogram')

            # Set the x-axis limits to reflect the duration per window
            ax.set_xlim(0, duration_per_window)

            # Set the figure size to 300x300
            fig.set_size_inches(3, 3)

            try:
                # Extract metadata from updated_signal_dictionary
                key = updated_signal_dictionary[i][0]
                median_peak = updated_signal_dictionary[i][3]
                median_std = updated_signal_dictionary[i][4]
                threshold = updated_signal_dictionary[i][5]

                # Generate a unique filename
                filename = f'spectrogram_{i}_{key}_{median_peak:.2f}_{median_std:.2f}_{threshold:.2f}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            except IndexError:
                # Handle IndexError if the index exceeds the dictionary length
                filename = f'spectrogram_{i}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

            # Save the spectrogram image to the folder
            save_path = os.path.join(folder_path, filename)
            plt.savefig(save_path)
            plt.close() 
            
            
    def save_PeakSspectrogramsType_2(self, selected_peak_windows, folder_path, figure_size=(3, 3)):
        """
        Generate and save spectrograms for given windows.

        Parameters:
            selected_peak_windows (list): List of signal windows to process.
            folder_path (str): Path to save the spectrogram images.
            figure_size (tuple): Dimensions of the saved spectrogram figure in inches (default is (3, 3)).
        """
        ADC_MAX_SAMPLE_FREQUENCY = 125000000  # Hz
        ADC_SAMPLE_DECIMATION = 64

        effective_sampling_frequency = ADC_MAX_SAMPLE_FREQUENCY / ADC_SAMPLE_DECIMATION
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i, window in enumerate(selected_peak_windows):
            # Create a new figure and axis
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot the spectrogram
            ax.specgram(window, Fs=effective_sampling_frequency, mode='psd', scale='dB', cmap='gray')

            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            ax.set_title('Spectrogram')

            # Set the figure size
            fig.set_size_inches(*figure_size)

            # Get the current date and time
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create a unique filename
            filename = f'spectrogram_{i}_{timestamp}.png'
            save_path = os.path.join(folder_path, filename)

            # Save the figure and close it
            plt.savefig(save_path)
            plt.close(fig)
            
    def extract_non_peak_windows(self,updated_signal_dictionary):
        """
        Extracts and selects non-peak windows from the updated signal dictionary.

        Returns:
            list: A list of randomly selected non-peak windows.
        """
        selected_non_peak_windows = []

        for i, element in enumerate(updated_signal_dictionary):
            # Access elements from updated_signal_dictionary
            key = element[0]  # Key
            filtered_signal = element[1]  # Filtered signal (Pandas Series)
            peaks = element[2]  # Detected peaks
            median_peak = element[3]  # Median peak value
            median_std = element[4]  # Median standard deviation
            threshold = element[5]  # Threshold value
            filtered_peaks = element[6]  # Filtered peaks

            # Determine window size
            if len(peaks) > 1:
                window_size = np.min(np.diff(peaks))  # Minimum distance between consecutive peaks
            else:
                window_size = len(filtered_signal)  # Entire signal length if only one peak is detected

            # Adjust window size if needed
            window_size *= 2

            # Create a list to store windows without the peak
            windows_without_peak = []

            # Extract windows without the median peak
            for k in range(0, len(filtered_signal), window_size):
                window = filtered_signal.iloc[k:k + window_size]  # Extract the window
                if median_peak not in window.values:  # Check if the median peak is not in the window
                    windows_without_peak.append(window)

            # Select a random window from windows_without_peak
            if windows_without_peak:
                selected_window = random.choice(windows_without_peak)
                selected_non_peak_windows.append(selected_window)

        return selected_non_peak_windows
    
    
    def save_NonPeakSspectrograms(self,selected_non_peak_windows, folder_path):
        """
        Saves spectrograms of non-peak windows to the specified folder.

        Args:
        selected_non_peak_windows (list): List of non-peak signal windows (e.g., pandas Series or numpy arrays).
        folder_path (str): Path to the folder where spectrogram images will be saved.
        adc_max_sample_frequency (int): Maximum ADC sampling frequency in Hz. Default is 125000000 Hz.
        adc_sample_decimation (int): ADC sampling decimation factor. Default is 64.
        """
        
        adc_max_sample_frequency=125000000
        adc_sample_decimation=64
        # Calculate the effective sampling frequency
        effective_sampling_frequency = adc_max_sample_frequency / adc_sample_decimation

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Loop through the non-peak windows
        for i, window in enumerate(selected_non_peak_windows):
            # Create a new figure and axis
            fig, ax = plt.subplots(figsize=(6, 6))
        
            # Plot the spectrogram
            ax.specgram(window, Fs=effective_sampling_frequency, mode='psd', scale='dB', cmap='gray')
        
            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            ax.set_title('Spectrogram')

            # Set the figure size to 300x300
            fig.set_size_inches(3, 3)
        
            # Generate a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate the file path and save the spectrogram
            filename = f'spectrogram_{i}_{timestamp}.png'
            save_path = os.path.join(folder_path, filename)
            plt.savefig(save_path)
        
            # Close the figure to avoid memory leaks
            plt.close(fig)
    
 

    
    


    
    