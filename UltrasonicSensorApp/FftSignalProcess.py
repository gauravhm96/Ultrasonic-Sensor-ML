import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class FftSignal:
    def __init__(self):
        self.folder_path = None

    def get_fft_data(self, file_path):
        try:
            # Open the file and read it line by line
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Clean the data by removing the first 16 elements from each row except the first one
            cleaned_lines = []
            for i, line in enumerate(lines):
                columns = line.split()
                if i == 0:
                    # Keep the first row completely
                    cleaned_lines.append(' '.join(columns))
                else:
                    cleaned_lines.append(' '.join(columns[16:]))

            # Join the cleaned lines into a string
            cleaned_data = "\n".join(cleaned_lines)

            # Read the cleaned data into pandas
            from io import StringIO
            signal_data = pd.read_csv(StringIO(cleaned_data), delimiter=r'\s+', header=None)
            return signal_data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def plot_data(self, signal_data):
            x_labels = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)

            #Get the data for y-axis (remove first row and use the rest)
            y_data = signal_data.iloc[1:, :]

            fig, axs = plt.subplots(1, 2, figsize=(10, 6))

            axs[0].plot(x_labels, y_data.values.T)  
            axs[0].set_title('Signal Data')  
            axs[0].set_xlabel('Frequency (Hz)') 
            axs[0].set_ylabel('Signal Value')   
            axs[0].grid(True) 
            
            for row in y_data.values:
                # Find peaks in the row (signal data)
                peaks, _ = find_peaks(row)

                # Plot the signal data
                axs[1].plot(x_labels, row, label="Signal Data")

                # Highlight the peaks
                axs[1].plot(x_labels[peaks], row[peaks], "x", color="red", label="Peaks")
            
            axs[1].set_title('Signal Data with Peaks')
            axs[1].set_xlabel('Frequency (Hz)')  
            axs[1].set_ylabel('Signal Value')    
            axs[1].grid(True)  

            plt.tight_layout()
            plt.show()

    def calculate_F2(self, signal_data):

        # Extract the first row for x-axis labels (frequencies)
        x_labels = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)

        # Get the data for y-axis (remove first row and use the rest)
        y_data = signal_data.iloc[1:, :]

        peak_frequencies = []  # List to store the peak frequencies for each signal

        # Extract peak frequencies for each signal row
        for row in y_data.values:
            # Find peaks in the row (signal data)
            peaks, _ = find_peaks(row)  # Find peaks in the signal row

            if len(peaks) > 0:
                # Find the index of the highest peak
                highest_peak_index = np.argmax(row[peaks])  # Index of the highest peak in the row
                peak_frequency = x_labels[peaks[highest_peak_index]]  # Frequency corresponding to that peak
                peak_frequencies.append(peak_frequency)
            else:
                peak_frequencies.append(None)  # No peaks found for this signal

        # Return the list of peak frequencies
        return peak_frequencies


    def calculate_F1(self,peak_frequencies):
        # Calculate the mean difference of F2 over 10 measurements (moving window)
        mean_differences = []
        for i in range(len(peak_frequencies) - 9):  # Subset the list for windows of 10
            window = peak_frequencies[i:i+10]  # Get the current window of 10 F2 values
            # Calculate differences between consecutive center frequencies in the window
            differences = np.diff(window)
            mean_diff = np.mean(differences)  # Calculate mean difference for this window
            mean_differences.append(mean_diff)
        
        # Calculate the average of all the mean differences
        F1 = np.mean(mean_differences)
        return F1
    
    def calculate_F3(self,peak_frequencies):
        # Calculate the variance of F2 over 10 measurements (moving window)
        variances = []
        for i in range(len(peak_frequencies) - 9):  # Subset the list for windows of 10
            window = peak_frequencies[i:i+10]  # Get the current window of 10 values
            var_f2 = np.var(window)  # Calculate variance for this window
            variances.append(var_f2)

        # Calculate the average variance over all windows
        F3 = np.mean(variances)
        return F3
    
    def calculate_F4(self,signal_data):
        frequency_bandwidth = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)
        
            # Check if frequency_bandwidth is a pandas Series or a list and handle accordingly
        if isinstance(frequency_bandwidth, pd.Series):
            # It's a pandas Series, so you can safely access the first and last elements
            lower_limit = frequency_bandwidth.iloc[0]  # First frequency
            upper_limit = frequency_bandwidth.iloc[-1]  # Last frequency
        else:
            # If it's not a pandas Series, convert it to a list and get the values
            frequency_bandwidth = list(frequency_bandwidth)
            lower_limit = frequency_bandwidth[0]
            upper_limit = frequency_bandwidth[-1]

        filtered_data = signal_data.iloc[1:, :]  # Ignore the first row (which is frequency values)

        # Find peaks in the filtered data (we will check each column's signal data for peaks)
        peaks_count = 0
        for column in filtered_data.columns:
            signal_column_frequencies = frequency_bandwidth

            # Using find_peaks from scipy to detect peaks
            peaks, _ = find_peaks(filtered_data[column].values)

            for peak_index in peaks:
                if peak_index < len(signal_column_frequencies):
                   peak_frequency = signal_column_frequencies[peak_index]
                   if lower_limit <= peak_frequency <= upper_limit:
                      peaks_count += 1
        return peaks_count

    def calculate_F5(self,signal_data):
        # Frequency bandwidth (first row as frequency labels)
        frequency_bandwidth = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)
        # List to store peak frequencies for each measurement
        peak_frequencies = []   

        filtered_data = signal_data.iloc[1:, :]  # Ignore the first row (which is frequency values)

        # Iterate over each column (signal data for each measurement)
        for column in filtered_data.columns:

            # Extract the signal column values
            signal_column = filtered_data[column].values
            
            # Using find_peaks from scipy to detect peaks
            peaks, _ = find_peaks(signal_column)

            # Get the frequencies corresponding to the detected peaks
            peak_frequencies_for_column = signal_column[peaks]
            peak_frequencies.append(peak_frequencies_for_column)

        # Convert peak_frequencies to a 2D numpy array (rows are measurements, columns are peaks)
        peak_frequencies_array = [np.array(peaks) for peaks in peak_frequencies]

        # Calculate the mean frequency for a moving window of 10 measurements
        F5_values = []
        window_size = 10

        for i in range(len(peak_frequencies_array) - window_size + 1):
            # Extract the peak frequencies within the current window
            window_peak_frequencies = peak_frequencies_array[i:i + window_size]

             # Flatten the window's peak frequencies
            flat_peak_frequencies = np.concatenate(
                [freqs for freqs in window_peak_frequencies if len(freqs) > 0]
            )

            # Calculate the mean if there are frequencies, else append NaN
            if len(flat_peak_frequencies) > 0:
                mean_frequency = np.mean(flat_peak_frequencies)
            else:
                mean_frequency = np.nan

            F5_values.append(mean_frequency)

        return np.array(F5_values)

    def calculate_F6(self,signal_data):
        # Frequency bandwidth (first row as frequency labels)
        frequency_bandwidth = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)
        # List to store peak frequencies for each measurement
        peak_frequencies = []   

        filtered_data = signal_data.iloc[1:, :]  # Ignore the first row (which is frequency values)

        # Iterate over each column (signal data for each measurement)
        for column in filtered_data.columns:

            # Extract the signal column values
            signal_column = filtered_data[column].values
            
            # Using find_peaks from scipy to detect peaks
            peaks, _ = find_peaks(signal_column)

            # Get the frequencies corresponding to the detected peaks
            peak_frequencies_for_column = signal_column[peaks]
            peak_frequencies.append(peak_frequencies_for_column)

        # Convert peak_frequencies to a 2D numpy array (rows are measurements, columns are peaks)
        peak_frequencies_array = [np.array(peaks) for peaks in peak_frequencies]

        # Calculate the variance of peak frequencies for a moving window of 10 measurements
        F6_values = []
        window_size = 10

        for i in range(len(peak_frequencies_array) - window_size + 1):
            # Extract the peak frequencies within the current window
            window_peak_frequencies = peak_frequencies_array[i:i + window_size]

            # Flatten the window's peak frequencies
            flat_peak_frequencies = np.concatenate(
                [freqs for freqs in window_peak_frequencies if len(freqs) > 0]
            )

            # Calculate the variance if there are frequencies, else append NaN
            if len(flat_peak_frequencies) > 0:
                variance_frequency = np.var(flat_peak_frequencies)
            else:
                variance_frequency = np.nan

            F6_values.append(variance_frequency)

        return np.array(F6_values)
    
    def calculate_F7(self,signal_data):
        # Frequency bandwidth (first row as frequency labels)
        frequency_bandwidth = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)
        # List to store peak frequencies for each measurement
        mean_frequencies = []   

        filtered_data = signal_data.iloc[1:, :]  # Ignore the first row (which is frequency values)

        # Iterate over each column (signal data for each measurement)
        for column in filtered_data.columns:

            # Extract the signal column values
            signal_column = filtered_data[column].values
            
            # Using find_peaks from scipy to detect peaks
            peaks, _ = find_peaks(signal_column)

            # Check if there are enough peaks to select the left, center (max peak), and right peaks
            if len(peaks) >= 3:
                # Get the indices for the left, center, and right peaks (center being the max peak)
                peak_values = signal_column[peaks]
                max_peak_index = np.argmax(peak_values)

                # Ensure we have valid left and right peaks around the center
                left_index = peaks[max_peak_index - 1] if max_peak_index > 0 else None
                right_index = peaks[max_peak_index + 1] if max_peak_index < len(peaks) - 1 else None

                if left_index is not None and right_index is not None:
                    if 0 <= left_index < len(frequency_bandwidth) and 0 <= right_index < len(frequency_bandwidth):
                        left_frequency = frequency_bandwidth[left_index]
                        center_frequency = frequency_bandwidth[peaks[max_peak_index]]
                        right_frequency = frequency_bandwidth[right_index]
                        mean_frequency = np.mean([left_frequency, center_frequency, right_frequency])
                
            # Append the mean frequency for this measurement
            mean_frequencies.append(mean_frequency)
        # Return the array of mean frequencies
        return np.array(mean_frequencies)

if __name__ == "__main__":
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_40_Hand.txt'
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_70.txt'
    
    folder_path = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_40_Wall.txt'
    #folder_path = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_40_Hand.txt'
    
    fft_signal = FftSignal()

    if folder_path is not None:
       signal_data = fft_signal.get_fft_data(folder_path)

       #fft_signal.plot_data(signal_data)
       print("MATLAB-style FFT plots saved successfully.")

       peak_frequencies = fft_signal.calculate_F2(signal_data)
       #print("Peak Frequencies:", peak_frequencies)

       F1 = fft_signal.calculate_F1(peak_frequencies)
       #print(f"Calculated Feature 1 Mean(F1): {F1}")

       F3 = fft_signal.calculate_F3(peak_frequencies)
       #print(f"Calculated Feature 3 Variance(F3): {F3}")

       F4 = fft_signal.calculate_F4(signal_data)
       #print(f"Calculated Feature 4 Peaks(F4): {F4}")

       F5 = fft_signal.calculate_F5(signal_data)
       #print(f"Calculated Feature 5 Mean(F5) based on F4: {F5}")

       F6 = fft_signal.calculate_F6(signal_data)
       #print(f"Calculated Feature 6 Variance(F5) based on F4: {F6}")

       F7 = fft_signal.calculate_F7(signal_data)
       print(f"Calculated Feature 7 Mean Frequency(F7) around Max Peak: {F7}")

    #Mean Difference of Center Frequencies (F1)
    #Center Frequency (F2)
    #Mean of Center Frequency (F3)
    #Variance of Center Frequency (F4)
    #Number of Peaks (F5)
    #Mean of Peak Count (F6
    #Variance of Peak Count (F7)
    #Mean Frequency Distance of Peaks (F8):
    #Mean of Frequency Distance (F9)
    #Variance of Frequency Distance (F10)
    #Feature 11(Skewness)
    #eature 12(Kurtosis)