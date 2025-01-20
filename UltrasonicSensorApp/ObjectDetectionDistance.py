import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,savgol_filter, find_peaks, peak_prominences, peak_widths

class ObjectDetectionDistance:
    def __init__(self, file_path):
        self.file_path = file_path  # Store the file path
    
    def calculate_distance(self):
        """Function to read the signal data from the file."""
        try:
            # Attempt to read the signal data from the file
            signal_data = pd.read_csv(self.file_path, delimiter='\t', header=None)
            return signal_data
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def process_signal_data(self, signal_data):
        """Process the signal data and find peaks without plotting."""
        df = pd.DataFrame(signal_data)
        signal_dictionary = []

        # Process each signal row and find peaks
        for i in range(df.shape[0]):
            # Ignoring the first 16 columns (based on your previous code)
            signal = df.iloc[i].iloc[16:]
            absolute_signal = signal.abs()  # Taking the absolute values

            # Calculate mean and standard deviation of the absolute signal
            mean_signal = np.mean(absolute_signal)
            std_signal = np.std(absolute_signal)

            # Set prominence threshold dynamically based on mean and standard deviation
            prominence_threshold = mean_signal + 3 * std_signal  

            # Find the peaks
            peaks, _ = find_peaks(absolute_signal.values[:], prominence=prominence_threshold)

            # Store the processed information
            key = f'S{i+1}'
            signal_dictionary.append((key, absolute_signal, peaks))

        # Return the dictionary containing signal information
        return signal_dictionary
    
    def annotate_real_peaks(self, signal_dictionary):
        """Annotates the real peaks and updates the signal dictionary with the information."""
        updated_signal_dictionary = []

        for key, signal, peaks in signal_dictionary:
            peak_values = signal.values[peaks]
            median_peak = np.median(peak_values)
            median_std = np.std(peak_values)

            # Calculate threshold multiplier k based on Z-score
            z_scores = (peak_values - median_peak) / median_std
            k = np.median(np.abs(z_scores))  # Use the average absolute Z-score as k

            threshold = median_peak + k * median_std

            # Filter peaks based on threshold
            filtered_peaks = [peak for peak in peak_values if peak <= threshold]

            # Store the annotated signal information in the updated dictionary
            updated_signal_dictionary.append({
                'key': key,
                'signal': signal,
                'peaks': peaks,
                'median_peak': median_peak,
                'median_std': median_std,
                'threshold': threshold,
                'filtered_peaks': filtered_peaks
            })

        # Store the updated signal dictionary
        self.updated_signal_dictionary = updated_signal_dictionary
        return updated_signal_dictionary
    
    import matplotlib.pyplot as plt
import numpy as np

def process_signal_plot(updated_signal_dictionary):
    """Process and prepare the signal data for plotting, without actually plotting it."""
    
    # Extract all threshold values from updated_signal_dictionary
    threshold_values = [element[5] for element in updated_signal_dictionary]  # Assuming threshold is stored at index 5

    # Calculate the overall threshold as the median of all threshold values
    overall_threshold = np.median(threshold_values)

    # Prepare figure and axes for two subplots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot total signal with overall threshold line (without showing it)
    for key, signal, peaks, median_peak, median_std, threshold, filtered_peaks in updated_signal_dictionary:
        axs[0].plot(signal.values, label=f'Signal {key}')

    axs[0].axhline(y=overall_threshold, color='red', linestyle='-', label=f'Amplitude(Peak): {overall_threshold:.2f}')
    axs[0].text(len(signal) - len(signal) // 20, overall_threshold + 0.05, f'Amplitude(Peak): {overall_threshold:.2f}', color='red', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Find the time index corresponding to the overall_threshold
    overall_threshold_index = np.argmax(signal.values > overall_threshold)

    # Annotate x-axis line with corresponding value
    axs[0].axvline(x=overall_threshold_index, color='blue', linestyle='-', label=f'Time(us): {overall_threshold_index}')
    axs[0].text(overall_threshold_index, overall_threshold - 0.05, f'Time(us): {overall_threshold_index}', color='blue', fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    axs[0].set_xlabel('Time (us)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Absolute Signal (With Noise)')
    axs[0].grid(True)

    # Plot total signal with values above the overall threshold eliminated (without showing it)
    for key, signal, peaks, median_peak, median_std, threshold, filtered_peaks in updated_signal_dictionary:
        filtered_signal = signal.copy()
        filtered_signal[filtered_signal > overall_threshold] = 0
        axs[1].plot(filtered_signal.values, label=f'Signal {key}')

    axs[1].axhline(y=overall_threshold, color='red', linestyle='-', label=f'Amplitude(Peak): {overall_threshold:.2f}')
    axs[1].text(len(signal) - len(signal) // 20, overall_threshold + 0.05, f'Amplitude(Peak): {overall_threshold:.2f}', color='red', fontsize=12, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Annotate x-axis line with corresponding value
    axs[1].axvline(x=overall_threshold_index, color='blue', linestyle='-', label=f'Time(us): {overall_threshold_index}')
    axs[1].text(overall_threshold_index, overall_threshold - 0.05, f'Time(us): {overall_threshold_index}', color='blue', fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    axs[1].set_xlabel('Time (us)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Absolute Signal (Without Noise)')
    axs[1].grid(True)

    # Add a main title to the figure
    fig.suptitle('Signal Processing (Noise Filtering)', fontsize=16)

    # Layout adjustment (no actual plot call)
    plt.tight_layout()

    # Return the prepared figure object without plotting it
    return fig, axs


def process_signal(filtered_signal):
    """
    Processes the filtered signal by applying Savitzky-Golay filtering, detecting peaks, and 
    filtering peaks based on prominence and width criteria. Returns key processed data.

    Parameters:
    - filtered_signal: A Pandas Series or DataFrame containing the filtered signal.

    Returns:
    - smoothed_df: A DataFrame containing the smoothed signal.
    - significant_peaks: Indices of the peaks above the prominence threshold.
    - actual_peaks: Indices of the final selected peaks, filtered by prominence and width.
    - max_peak_index: Index of the maximum amplitude peak among the actual peaks.
    """
    
    # Apply Savitzky-Golay filtering
    window_length = 4  # Adjust the window length as needed
    polyorder = 2  # Adjust the polynomial order as needed
    smoothed_signal = savgol_filter(filtered_signal.values, window_length=window_length, polyorder=polyorder)

    # Convert smoothed_signal to a DataFrame
    smoothed_df = pd.DataFrame({'Index': filtered_signal.index, 'Smoothed Amplitude': smoothed_signal})

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

    return smoothed_df, significant_peaks, actual_peaks, max_peak_index
