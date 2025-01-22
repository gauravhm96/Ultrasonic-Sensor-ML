# signal_processor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_prominences, peak_widths


class SignalProcessor:
    def __init__(self):
        self.file_path = None
        self.signal_data = None
        self.signal_dictionary = []
        self.updated_signal_dictionary = []
        self.threshold_info = {}
        self.signal_correction_info = {}  
        self.distance_info = {}
      
    def set_file_path(self, file_path):
        self.file_path = file_path    

    def load_signal_data(self):
        try:
            self.signal_data = pd.read_csv(self.file_path, delimiter="\t", header=None)
            print("Signal data successfully loaded.")
            return self.signal_data
        except Exception as e:
            print(f"An error occurred while loading the signal data: {e}")
            return None

    def analyze_raw_signals(self):
        try:
            df = pd.DataFrame(self.signal_data)
            
            for i in range(df.shape[0]):
                signal = df.iloc[i].iloc[16:]
                absolute_signal = signal.abs()

                # Calculate mean and standard deviation
                mean_signal = np.mean(absolute_signal)
                std_signal = np.std(absolute_signal)

                # Dynamic threshold for peak detection
                prominence_threshold = mean_signal + 3 * std_signal
                peaks, _ = find_peaks(
                    absolute_signal.values[:], prominence=prominence_threshold
                )

                # Store signal information
                key = f"S{i+1}"
                self.signal_dictionary.append((key, absolute_signal, peaks))

            print("Signals analyzed successfully.")
        except Exception as e:
            print(f"An error occurred during signal analysis: {e}")
    
    def annotate_real_peaks(self):
        try:
            for key, signal, peaks in self.signal_dictionary:
                peak_values = signal.values[peaks]
                median_peak = np.median(peak_values)
                median_std = np.std(peak_values)

                # Calculate threshold multiplier k based on Z-score
                z_scores = (peak_values - median_peak) / median_std
                k = np.median(np.abs(z_scores))  # Use the average absolute Z-score as k

                threshold = median_peak + k * median_std

                # Filter peaks based on threshold
                filtered_peaks = [peak for peak in peaks if signal.values[peak] <= threshold]

                # Store the updated signal information
                self.updated_signal_dictionary.append((key, signal, peaks, median_peak, median_std, threshold, filtered_peaks))

            print("Real peaks annotated successfully.")
        except Exception as e:
            print(f"An error occurred while annotating the real peaks: {e}")


    def NoiseFiltering(self):
        # Extract all threshold values from updated_signal_dictionary (threshold is at index 5)
        threshold_values = [element[5] for element in self.updated_signal_dictionary]
        
        # Calculate the overall threshold as the median of all threshold values
        overall_threshold = np.median(threshold_values)
        
        # Now using the last signal to calculate the overall threshold index
        signal = self.updated_signal_dictionary[-1][1] # Access the signal part from the last tuple
        
        # Filter the signal: set values above overall_threshold to 0
        filtered_signal = signal.copy()
        filtered_signal[filtered_signal > overall_threshold] = 0
        
        # Find the index where the signal exceeds the overall threshold
        overall_threshold_index = np.argmax(signal.values > overall_threshold)
        
        self.threshold_info = {
        "threshold_values": threshold_values,
        "overall_threshold": overall_threshold,
        "overall_threshold_index": overall_threshold_index,
        "filtered_signal": filtered_signal
        }
         
    def SignalCorrection(self):
        # Apply Savitzky-Golay filtering
        window_length = 4  # Adjust the window length as needed
        polyorder = 2  # Adjust the polynomial order as needed
        smoothed_signal = savgol_filter(self.threshold_info["filtered_signal"].values, window_length=window_length, polyorder=polyorder)

        # Convert smoothed_signal to a DataFrame
        smoothed_df = pd.DataFrame({'Index': self.threshold_info["filtered_signal"].index, 'Smoothed Amplitude': smoothed_signal})

        # Find peaks from the smoothed signal
        peaks, _ = find_peaks(smoothed_df['Smoothed Amplitude'])

        # Calculate the mean and standard deviation of the smoothed signal
        mean_signal = smoothed_df['Smoothed Amplitude'].mean()
        std_signal = smoothed_df['Smoothed Amplitude'].std()

        # Set prominence threshold dynamically based on mean and standard deviation
        prominence_threshold = mean_signal + 3 * std_signal  # Adjust the multiplier as needed

        # Find peaks above the dynamically calculated prominence threshold
        prominences = peak_prominences(smoothed_df['Smoothed Amplitude'], peaks)[0]
        significant_peaks = peaks[prominences > prominence_threshold]

        # Find widths of the peaks
        widths, _, _, _ = peak_widths(smoothed_df['Smoothed Amplitude'], significant_peaks)

        # Calculate the median width of the peaks
        median_width = np.median(widths)

        # Set the minimum width threshold dynamically
        min_width = median_width  # You can also use mean or another percentile

        # Filter out peaks based on width
        actual_peaks = significant_peaks[widths > min_width]

        # Find the index of the maximum peak among the actual peaks
        max_peak_index = actual_peaks[np.argmax(smoothed_df['Smoothed Amplitude'][actual_peaks])]

        # Store the results in threshold_info dictionary
        self.signal_correction_info = {
            "smoothed_signal": smoothed_signal,
            "smoothed_df": smoothed_df,
            "peaks": peaks,
            "significant_peaks": significant_peaks,
            "actual_peaks": actual_peaks,
            "max_peak_index": max_peak_index,
            "max_peak_amplitude": smoothed_df['Smoothed Amplitude'][max_peak_index]
        }
        
    def PrintInputdata(self):
        return self.signal_data.head()
    
    def PlotRawSignal(self):
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # Plot all the signals
        df = pd.DataFrame(self.signal_data)  # Ensure the signal data is in a DataFrame
        
        # List to hold plot data for return
        plot_data = []
        
        for i in range(df.shape[0]):
            signal = df.iloc[i].iloc[16:]  # Ignoring the first 16 columns
            axs[0].plot(signal.values[:])

        # Adding labels and title to the first subplot
        axs[0].grid(True)
        axs[0].set_xlabel('Time (samples)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Signal Data')

        # Plot all the signals with absolute values
        for i in range(df.shape[0]):
            signal = df.iloc[i].iloc[16:]  # Ignoring the first 16 columns
            absolute_signal = signal.abs()  # Taking the absolute values

            # Calculate mean and standard deviation of the absolute signal
            mean_signal = np.mean(absolute_signal)
            std_signal = np.std(absolute_signal)

            # Set prominence threshold dynamically based on mean and standard deviation
            prominence_threshold = mean_signal + 3 * std_signal  

            peaks, _ = find_peaks(absolute_signal.values[:], prominence=prominence_threshold)

            key = f'S{i+1}'
            self.signal_dictionary.append((key, absolute_signal, peaks))

            axs[1].plot(absolute_signal.values[:])
            axs[1].plot(peaks, absolute_signal.values[:][peaks], 'x', color='green')  # Plotting the peaks for visualization
        # Adding labels and title to the second subplot
        axs[1].grid(True)
        axs[1].set_xlabel('Time (samples)')
        axs[1].set_ylabel('Absolute Amplitude')
        axs[1].set_title('Absolute Value of Signals With Peaks')

        # Add a main title
        fig.suptitle('Raw Data Captured from Ultrasonic Sensor', fontsize=16)

        # Adjust layout
        plt.tight_layout()
        
        return fig, axs
    
    def PlotNoiseFilteredSignal(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        # Plot original signal with threshold line
        for key, signal, peaks, median_peak, median_std, threshold, filtered_peaks in self.updated_signal_dictionary:
            axs[0].plot(signal.values, label=f'Signal {key}')
        
        # Plot overall threshold
        threshold_values = [element[5] for element in self.updated_signal_dictionary]
        overall_threshold = np.median(threshold_values)
        axs[0].axhline(y=overall_threshold, color='red', linestyle='-', label=f'Amplitude(Peak): {overall_threshold:.2f}')
        
        # Annotate with threshold values
        axs[0].text(len(signal) - len(signal) // 20, overall_threshold + 0.05, f'Amplitude(Peak): {overall_threshold:.2f}', color='red', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        # Find the time index corresponding to the overall threshold
        overall_threshold_index = np.argmax(signal.values > overall_threshold)
        axs[0].axvline(x=overall_threshold_index, color='blue', linestyle='-', label=f'Time(us): {overall_threshold_index}')
        axs[0].text(overall_threshold_index, overall_threshold - 0.05, f'Time(us): {overall_threshold_index}', color='blue', fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                
        axs[0].set_xlabel('Time (us)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Absolute Signal (With Noise)')
        axs[0].grid(True)
        
        # Plot filtered signal (noise removed)
        for key, signal, peaks, median_peak, median_std, threshold, filtered_peaks in self.updated_signal_dictionary:
            filtered_signal = signal.copy()
            filtered_signal[filtered_signal > overall_threshold] = 0
            axs[1].plot(filtered_signal.values, label=f'Signal {key}')
            
        axs[1].axhline(y=overall_threshold, color='red', linestyle='-', label=f'Amplitude(Peak): {overall_threshold:.2f}')
        axs[1].text(len(signal) - len(signal) // 20, overall_threshold + 0.05, f'Amplitude(Peak): {overall_threshold:.2f}', color='red', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        axs[1].axvline(x=overall_threshold_index, color='blue', linestyle='-', label=f'Time(us): {overall_threshold_index}')
        axs[1].text(overall_threshold_index, overall_threshold - 0.05, f'Time(us): {overall_threshold_index}', color='blue', fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        axs[1].set_xlabel('Time (us)')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_title('Absolute Signal (Without Noise)')
        axs[1].grid(True)
        
        fig.suptitle('Signal Processing (Noise Filtering)', fontsize=16)
        plt.tight_layout()
        
        return fig,axs
    
    def PlotSignalCorrection(self):
        
        # Get signal correction info from the thresholding process
        smoothed_signal = self.signal_correction_info.get('smoothed_signal', None)
        smoothed_df = self.signal_correction_info.get('smoothed_df', None)
        significant_peaks = self.signal_correction_info.get('significant_peaks', None)
        actual_peaks = self.signal_correction_info.get('actual_peaks', None)
        max_peak_index = self.signal_correction_info.get('max_peak_index', None)

        if smoothed_signal is not None and smoothed_df is not None:
            # Plot all the figures as subplots
            fig, axs = plt.subplots(1, 4, figsize=(36, 12))
                         
            # Plot smoothed signal
            axs[0].plot(smoothed_df['Index'], smoothed_df['Smoothed Amplitude'], label='Smoothed Signal', color='red')
            axs[0].set_xlabel('Time(us)')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title('Smoothed Signal (Savitzky-Golay Filtering)')
            axs[0].legend()
            axs[0].grid(True)
                        
            # Plot smoothed signal with peaks
            axs[1].plot(smoothed_df['Index'], smoothed_df['Smoothed Amplitude'], label='Smoothed Signal', color='red')
            axs[1].plot(smoothed_df['Index'][significant_peaks], smoothed_df['Smoothed Amplitude'][significant_peaks], 'x', color='green', label='Peaks')
            axs[1].set_xlabel('Time(us)')
            axs[1].set_ylabel('Amplitude')
            axs[1].set_title('Smoothed Signal with Peaks')
            axs[1].legend()
            axs[1].grid(True)
                        
            # Plot smoothed signal with actual peaks
            axs[2].plot(smoothed_df['Index'], smoothed_df['Smoothed Amplitude'], label='Smoothed Signal', color='red')
            axs[2].plot(smoothed_df['Index'][actual_peaks], smoothed_df['Smoothed Amplitude'][actual_peaks], 'x', color='green', label='Actual Peaks')
            axs[2].set_xlabel('Time(us)')
            axs[2].set_ylabel('Amplitude')
            axs[2].set_title('Smoothed Signal with Actual Peaks')
            axs[2].legend()
            axs[2].grid(True)

            # Plot original and smoothed signals with maximum actual peak
            axs[3].plot(smoothed_df['Index'], smoothed_df['Smoothed Amplitude'], label='Smoothed Signal', color='red')
            axs[3].plot(smoothed_df['Index'][max_peak_index], smoothed_df['Smoothed Amplitude'][max_peak_index], 'x', color='green', label='Maximum Actual Peak')
            axs[3].set_xlabel('Time(us)')
            axs[3].set_ylabel('Amplitude')
            axs[3].set_title('Original and Smoothed Signals with Maximum Actual Peak')
            axs[3].legend()
                        
            max_peak_amplitude = smoothed_df['Smoothed Amplitude'][max_peak_index]
            axs[3].axhline(y=max_peak_amplitude, color='blue', linestyle='-')
            axs[3].text(max_peak_index, max_peak_amplitude + 0.5, f'Peak Amplitude: {max_peak_amplitude:.2f}', color='blue', fontsize=10, ha='right', va='bottom')

            # Highlight corresponding x-axis value
            axs[3].axvline(x=max_peak_index, color='blue', linestyle='-')
            axs[3].text(max_peak_index, 0.5, f'Time(us): {max_peak_index}', color='blue', fontsize=10, ha='right', va='bottom')

            axs[3].grid(True)

            # Add a main title
            fig.suptitle('Signal Correction', fontsize=16)
            plt.tight_layout()
        return fig, axs
       
    def Render_Output(self):
                                       
        user_input = input("Do you want to view the output information (real peaks, threshold, etc.)? (yes/no): ").strip().lower()
        
        if user_input == 'yes':
           try:  
                self.print_output_info()
                print("Annotated Peaks Successful.")
           except Exception as e:
                print(f"An error occurred while rendering peaks: {e}")
        elif user_input == 'no':
            print("Render print processed peaks skipped.")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            
        user_input = input("Do you want to view Calculated Data? (yes/no): ").strip().lower()
        
        if user_input == 'yes':
            try:
                if self.distance_info:
                    if self.distance_info.get("TOF") is not None:
                        print("=== Distance Calculation Results ===")
                        print("ADC Sample Frequency: {:.2f} Hz".format(self.distance_info["ADC_SAMPLE_FREQUENCY"]))
                        print("Time Interval:", self.distance_info["time_interval"], "seconds")
                        print("Time of Flight (TOF):", self.distance_info["TOF"], "seconds")
                        print("Distance: {:.2f} meters".format(self.distance_info["distance"]))
                        print("Calculations done successfully.")
                    else:
                        print("Error in Distance Calculation: max_peak_index not found!")
            except Exception as e:
                print(f"An error occurred while calculating data: {e}")
        elif user_input == 'no':
            print("View Calculated Data Skipped.")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            
                
    def Calculate_Distance(self):
        # Constants
        ADC_MAX_SAMPLE_FREQUENCY = 125000000  # Hz
        ADC_SAMPLE_DECIMATION = 64
        ADC_SAMPLE_FREQUENCY = ADC_MAX_SAMPLE_FREQUENCY / ADC_SAMPLE_DECIMATION  # Hz
        ADC_SAMPLE_TIME = 8  # ns
        ADC_SAMPLE_TIME_NS = ADC_SAMPLE_DECIMATION * ADC_SAMPLE_TIME  # ns
        ADC_START_DELAY_US = int(0.30 * 2 * 1e6 / 343.2)  # µs
        ADC_BUFFER_SIZE = 16384
        ADC_BUFFER_DELAY_US = int((ADC_BUFFER_SIZE * ADC_SAMPLE_TIME_NS) / 1e3)  # µs
        ADC_MID_US = ADC_START_DELAY_US + (ADC_BUFFER_DELAY_US // 2)  # µs

        speed_of_sound = 343  # in meters per second

        # Calculate the time interval based on the ADC sampling frequency
        time_interval = 1 / ADC_SAMPLE_FREQUENCY  # in seconds

        max_peak_index = self.signal_correction_info.get('max_peak_index', None)
        # Calculate Time of Flight (TOF) using max_peak_index and considering ADC delays
        TOF = max_peak_index * time_interval + ADC_START_DELAY_US / 1e6  # Time of flight in seconds

        # Convert TOF to distance
        distance = TOF * speed_of_sound / 2  # Distance in meters

        # Store results in dictionary for later use
        self.distance_info = {
            "ADC_SAMPLE_FREQUENCY": ADC_SAMPLE_FREQUENCY,
            "time_interval": time_interval,
            "TOF": TOF,
            "distance": distance
        }
        return self.distance_info

    def print_output_info(self):
        if self.updated_signal_dictionary:
            print("\nOutput Information:")
            for key, signal, peaks, median_peak, median_std, threshold, filtered_peaks in self.updated_signal_dictionary:
                print(f"\nTop Peaks in signal {key}: {signal.values[peaks]}")
                print(f"Real peak value for signal {key}: {median_peak}")
                print(f"Median Std for signal {key}: {median_std}")
                print(f"Threshold for signal {key}: {threshold}")
                print(f"Filtered Peaks for signal {key}: {filtered_peaks}")
                            
    def reset(self):
        self.signal_data = None
        self.signal_dictionary = []
        self.updated_signal_dictionary = []
        self.threshold_info = {}
        self.signal_correction_info = {}
        self.distance_info = {}  
        print("SignalProcessor has been reset.")


#if __name__ == "__main__":
    #processor.Render_Output()
    #processor.reset()
