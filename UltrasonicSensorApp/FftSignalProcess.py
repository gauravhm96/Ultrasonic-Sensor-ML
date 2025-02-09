import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.special import entr
from scipy.stats import entropy
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import get_window
import matplotlib.pyplot as plt


class FftSignal:
    def __init__(self):
        self.folder_path = None
    
    def get_fft_data(self, file_path):
        
        try:            
            raw_data = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    columns = line.strip().split('\t')  # Split by tab delimiter
                    if i > 0:  # For all rows except the first one
                        columns = columns[16:]  # Remove the first 16 elements
                    raw_data.append('\t'.join(columns))  # Join back the cleaned row
            # Join the cleaned lines into a string
            amplitude = "\n".join(raw_data)

            # Read the cleaned data into pandas
            from io import StringIO
            signal_data = pd.read_csv(StringIO(amplitude), delimiter='\t', header=None)

            # Add padding to the signal data
            return signal_data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def plot_raw_data(self,frequencyspectrum,amplitudebuffer):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(amplitudebuffer.shape[0]):  # Iterate through rows in amplitude buffer
            ax.plot(frequencyspectrum, amplitudebuffer.iloc[i, :], label=f'Time {i+1}')
        ax.set_title("Frequency Spectrum vs Amplitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        
        plt.show()
            
    def save_signal_data(self, signal_data, output_file_path):
        try:
            signal_data.to_csv(output_file_path, index=False, header=False, sep='\t')
            print(f"Signal data saved successfully to {output_file_path}")
        except Exception as e:
            print(f"Error saving signal data: {e}")
            
    def plot_data(self, X_axis, Y_axis):
        fig, axes = plt.subplots(4, 4, figsize=(15, 12))
        axes = axes.flatten() 
        
        for idx, ax in enumerate(axes):
            if idx < len(Y_axis):
               if isinstance(Y_axis[idx], pd.DataFrame) and len(Y_axis[idx].shape) == 2:
                  ax.plot(X_axis, Y_axis[idx].T)
                  ax.set_title(f"Plot {idx+1} - Test")
               else:
                  ax.plot(X_axis, Y_axis[idx])
                  ax.set_title(f"Plot {idx+1} - Just")  
               ax.set_xlabel("Frequency (Hz)")
               ax.set_ylabel("Amplitude")
            else:
               ax.axis('off')  # Turn off unused subplots if less than 16 plots
        plt.tight_layout()
        plt.show()   


class ExtractFeaturesFFT: 
    def __init__(self):
         pass

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
        
        get_bandwidth = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)

        peak_info_dict = {}
                        
        amplitude_buffer = signal_data.iloc[1:, :] 

        # Iterate over each column (signal data for each measurement)
        for frequency_index in amplitude_buffer.columns:

            # Extract the signal column values
            amplitude_array = amplitude_buffer[frequency_index].values
            peaks, _ = find_peaks(amplitude_array)
            
            if len(peaks) > 0:
            
                peak_values_array = amplitude_array[peaks]
            
                # Get the index of the maximum peak amplitude
                max_peak_index = np.argmax(peak_values_array)
            
                # Get the actual maximum peak amplitude using the index
                #max_peak_amplitude = peak_values_array[max_peak_index]
            
                frequency = get_bandwidth.iloc[frequency_index]
                #print(f"Frequency Index: {frequency_index}")
                #print(f"Freq: {frequency}")
                #print(f"Amplitudes Array: {amplitude_array}")
                #print(f"Peak Indices: {peaks}")
                #print(f"Peak Amplitudes: {peak_values_array}")
                #print(f"Max Peak Amplitude: {max_peak_amplitude}")
                
                peak_info_dict[frequency_index]= {
                                                   'Freq': frequency,
                                                   'Peak Amplitudes':peak_values_array
                                                 }
                
            else:
                print(f"No peaks found for frequency index: {frequency_index}")
            

        max_amplitude = -np.inf  # Initialize to a very small value    
        # Iterate through each frequency index in peak_info_dict
        for frequency_index, data in peak_info_dict.items():
            peak_values_array = data['Peak Amplitudes']

            # Find the maximum amplitude in peak_values_array
            current_max_amplitude = np.max(peak_values_array)
        
            # If the current max amplitude is larger than the previously found one
            if current_max_amplitude > max_amplitude:
                max_amplitude = current_max_amplitude
                corresponding_frequency = data['Freq']  # Save the corresponding frequency
        
        #print(max_amplitude,corresponding_frequency)
        
        max_peak_info = None
        max_peak_amplitude = -np.inf
            # Iterate over the peak_info_dict
        for frequency_index, peak_info in peak_info_dict.items():
            # Get the peak values array and corresponding frequencies
            peak_values_array = peak_info['Peak Amplitudes']

            # Find the maximum peak amplitude in the current peak_values_array
            current_max_peak_amplitude = np.max(peak_values_array)
            
            if current_max_peak_amplitude > max_peak_amplitude:
                # Get the index of the max peak
                max_peak_index = np.argmax(peak_values_array)
                
                 # Look for the left peak value, if none found, find the nearest peak
                left_peak_value = None
                for i in range(max_peak_index - 1, -1, -1):  # Check backward from max_peak_index
                    if peak_values_array[i] > 0:  # Look for a valid peak
                        left_peak_value = peak_values_array[i]
                        Leftfrequency = peak_info['Freq']  # Save the corresponding frequency
                        break
                    
                # Look for the right peak value, if none found, find the nearest peak
                right_peak_value = None
                for i in range(max_peak_index + 1, len(peak_values_array)):  # Check forward from max_peak_index
                    if peak_values_array[i] > 0:  # Look for a valid peak
                        right_peak_value = peak_values_array[i]
                        Righttfrequency = peak_info['Freq']  # Save the corresponding frequency
                        break
                
                max_peak_amplitude = current_max_peak_amplitude
                Maxfrequency = peak_info['Freq']  # Save the corresponding frequency
                
                max_peak_info = {
                'Left Peak': left_peak_value,
                'Left Freq': Leftfrequency,
                'Centre Peak': peak_values_array[max_peak_index],
                'Max Freq':Maxfrequency,
                'Right Peak': right_peak_value,
                'Right Freq': Righttfrequency,
                }
        print(max_peak_info)
        return max_peak_info
    
class getFFTSignalParameters:
    def __init__(self):
        pass
    
    def getfrequencyspectrum(self,signal_data):   
        frequencyspectrum = signal_data.iloc[0, :]  # First row as x-axis labels (frequency values)
        return frequencyspectrum
    
    def getamplitude(self,signal_data):
        amplitude_buffer = signal_data.iloc[1:, :] 
        return amplitude_buffer
    
    def getabsamplitude(self, signal_data):
        amplitude_buffer = signal_data.iloc[1:, :]
        magnitude_buffer = amplitude_buffer.abs()
        return magnitude_buffer
    
    def getFmax(self,frequencyspectrum):      
        fmax = frequencyspectrum.max()
        return fmax
        
    def getFmin(self,frequencyspectrum):
        fmin = frequencyspectrum.min()
        return fmin
        
    def getBW(self,fmax,fmin):
        BW = fmax - fmin
        return BW
        
    def getSamplingFrequency(self):
        ADC_MAX_SAMPLE_FREQUENCY = 125_000_000  # 125 MHz
        ADC_SAMPLE_DECIMATION = 64
        sampling_frequency = ADC_MAX_SAMPLE_FREQUENCY / ADC_SAMPLE_DECIMATION
        return sampling_frequency

    def getfreqfactor(self,sampling_frequency):
        window_width = 8192  # FFT points (window width)
        freq_factor = sampling_frequency / (window_width * 2)
        return freq_factor
    
    def getFreqresolution(self,BW):
        window_width = 8192  # FFT points (window width)
        freq_resolution = BW / window_width
        return freq_resolution
    
    def getBandPassFilterParameters(self,frequencyspectrum,MaxAmplitude,Totalpower):
        getsealed = getFFTSignalParameters()
        max_amplitude_value = np.max(MaxAmplitude)
        max_amplitude_indices = np.where(MaxAmplitude == max_amplitude_value)[0]
       
        # Find all indices where Totalpower reaches its maximum
        max_power_value = np.max(Totalpower)
        max_power_indices = np.where(Totalpower == max_power_value)[0]
        
        # Extract corresponding frequencies
        freq_amplitude_peaks = frequencyspectrum[max_amplitude_indices]
        freq_power_peaks = frequencyspectrum[max_power_indices]
        
        # Compute weighted center frequency (weighted by amplitudes and power at each peak)
        weighted_freq_amplitude = np.sum(freq_amplitude_peaks * MaxAmplitude[max_amplitude_indices]) / np.sum(MaxAmplitude[max_amplitude_indices])
        weighted_freq_power = np.sum(freq_power_peaks * Totalpower[max_power_indices]) / np.sum(Totalpower[max_power_indices])
        
        # Final center frequency: averaging both weighted results
        center_frequency_combined = (weighted_freq_amplitude + weighted_freq_power) / 2
        
        # Calculate -3 dB threshold (50% of max power)
        threshold = 0.5 * max_power_value

        # Find indices where power crosses the threshold
        indices_below_threshold = np.where(Totalpower >= threshold)[0]

        if len(indices_below_threshold) > 0:
            # Get the first and last frequency that meets the -3 dB criteria
            F1_3dB = frequencyspectrum[indices_below_threshold[0]]  # Lower cutoff
            F2_3dB = frequencyspectrum[indices_below_threshold[-1]]  # Upper cutoff
        else:
            # If no cutoff frequencies are found, set default values
            F1_3dB = None
            F2_3dB = None
            print("Warning: No cutoff frequencies found at -3 dB level.")
        alpha = 0.5  # Adjust this value as needed (e.g., between 0.1 and 0.3)
        F1 = F1_3dB - alpha * (F2_3dB - F1_3dB)
        F2 = F2_3dB + alpha * (F2_3dB - F1_3dB)
        
        F1_sealed = frequencyspectrum[np.abs(frequencyspectrum - F1).argmin()]
        F2_sealed = frequencyspectrum[np.abs(frequencyspectrum - F2).argmin()]
        center_frequency_sealed = frequencyspectrum[np.abs(frequencyspectrum - center_frequency_combined).argmin()]
        
        return center_frequency_sealed,F1_sealed,F2_sealed

    def seal_to_spectrum(self,value, frequencyspectrum):
        # Find the index of the closest frequency value in frequencyspectrum
        index = np.abs(frequencyspectrum - value).argmin()
        return frequencyspectrum[index]
        
class getFFTSignalFreqDomainFeatures:
    def __init__(self):
        self.features = {}
        
    def getentropy(self,amplitudebuffer):
        flattened_data = amplitudebuffer.to_numpy().flatten()  
        value, counts = np.unique(flattened_data, return_counts=True)
        probabilities = counts / len(flattened_data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def getwindowsize(self,entropy,frequency_resolution):
        if entropy < 7.5:  # Higher entropy means higher variation
            window_size = int(frequency_resolution * 2) 
        elif 7.5 <= entropy < 8.8 :
            window_size = int(frequency_resolution * 4)  # moderate window size
        elif entropy > 8.8:
            window_size = int(frequency_resolution * 10)  # larger window for stable signals
        return window_size
        
    def smoothenAmplitude(self,amplitudebuffer,window_size):
        # Apply a moving average convolution along the columns (axis=0) of the amplitudebuffer
        smoothed_amplitude = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='same'), 
            axis=0, 
            arr= np.array(amplitudebuffer)
        )
        smoothed_amplitude_df = pd.DataFrame(smoothed_amplitude, columns=amplitudebuffer.columns, index=amplitudebuffer.index)
        return smoothed_amplitude_df
    
    def getMeanAmplitude(self, amplitudebuffer):
        mean_amplitude = np.mean(amplitudebuffer, axis=0)
        self.features['mean_amplitude'] = mean_amplitude
        return mean_amplitude
    
    def getMaxAmplitude(self, amplitudebuffer):
        max_amplitude = np.max(amplitudebuffer, axis=0)
        self.features['max_amplitude'] = max_amplitude
        return max_amplitude
    
    def getPeakToPeak(self, amplitudebuffer):
        peak_to_peak = np.ptp(amplitudebuffer, axis=0)
        self.features['peak_to_peak'] = peak_to_peak
        return peak_to_peak
    
    def getRMSAmplitude(self, amplitudebuffer):
        rms_amplitude = np.sqrt(np.mean(amplitudebuffer**2, axis=0))
        self.features['rms_amplitude'] = rms_amplitude
        return rms_amplitude

    def getVariance(self, amplitudebuffer):
        variance = np.var(amplitudebuffer, axis=0)
        self.features['variance'] = variance
        return variance

    def getStdDev(self, amplitudebuffer):
        std_dev = np.std(amplitudebuffer, axis=0)
        self.features['std_dev'] = std_dev
        return std_dev

    def getSkewness(self, amplitudebuffer):
        skewness = skew(amplitudebuffer, axis=0)
        self.features['skewness'] = skewness
        return skewness

    def getKurtosis(self, amplitudebuffer):
        kurt = kurtosis(amplitudebuffer, axis=0)
        self.features['kurtosis'] = kurt
        return kurt
    
    def gettotalpower(self,amplitudebuffer):
        power = np.sum(amplitudebuffer**2, axis=0)
        self.features['power'] = power
        return power
    
    def getcrestfactor(self,amplitudebuffer):
        peak_amplitude = np.max(amplitudebuffer, axis=0)  # Maximum value (Peak)
        rms_amplitude = np.sqrt(np.mean(amplitudebuffer**2, axis=0))  # RMS value
        crest_factor = peak_amplitude / rms_amplitude
        self.features['Crest Factor'] = crest_factor
        return crest_factor
    
    def getformfactor(self,amplitudebuffer):
        mean_amplitude = np.mean(amplitudebuffer, axis=0)  # Mean value
        rms_amplitude = np.sqrt(np.mean(amplitudebuffer**2, axis=0))  # RMS value
        form_factor = rms_amplitude / mean_amplitude
        self.features['Form Factor'] = form_factor
        return form_factor
    
    def getpeaktomeanratio(self,amplitudebuffer):
        peak_amplitude = np.max(amplitudebuffer, axis=0)  # Peak amplitude
        mean_amplitude = np.mean(amplitudebuffer, axis=0)  # Mean amplitude
        pulse_indicator = peak_amplitude / mean_amplitude
        self.features['pulse_indicator'] = pulse_indicator
        return pulse_indicator
    
    def getmargin(self,amplitudebuffer):
        peak_amplitude = np.max(amplitudebuffer, axis=0)  # Peak amplitude
        noise_floor = np.min(amplitudebuffer, axis=0)  # Minimum amplitude (noise floor)
        margin = peak_amplitude - noise_floor  # Difference between peak and noise floor
        self.features['margin'] = margin
        return margin
    
    def getrelativepeakspectral(self,amplitudebuffer):
        relative_spectral_peak = (np.max(amplitudebuffer, axis=0)) / np.sum(amplitudebuffer, axis=0)
        self.features['relative_spectr8al_peak'] = relative_spectral_peak
        return relative_spectral_peak
    
    def applyHanningWindow(self, frequencyspectrum, amplitude_buffer,F1,F2):
        freq_range = np.array(frequencyspectrum)
        f1_index = np.searchsorted(freq_range, F1)
        f2_index = np.searchsorted(freq_range, F2)
        # Generate Hanning window for the passband
        window = np.zeros_like(freq_range, dtype=float)
        window[f1_index:f2_index + 1] = np.hanning(f2_index - f1_index + 1)
        
        amplitudebuffer_float = amplitude_buffer.astype(float)
        # Apply the Hanning window to the FFT data (amplitudebuffer)
        windowed_amplitudebuffer = amplitudebuffer_float.copy() 
        
        # Apply the window to each sample (row) of the amplitudebuffer
        for i in range(windowed_amplitudebuffer.shape[0]):  # Iterate over each row (sample)
           windowed_amplitudebuffer.iloc[i, :] *= window  # Apply the window across all frequency bins
           
        # Define the window function (you can try 'hamming', 'hann', 'blackmanharris', etc.)
        window_type = 'hann'  # Or 'hamming', 'blackmanharris', etc.
        window = get_window(window_type, amplitude_buffer.shape[1])

        # Apply the window to the amplitudebuffer and windowed_amplitudebuffer
        smoothed_windowed_amplitudebuffer = windowed_amplitudebuffer.multiply(window, axis=1)
        
        return smoothed_windowed_amplitudebuffer 
    
    def getfreqPCA(self,frequencyspectrum,amplitudebuffer):
        extract = getFFTSignalFreqDomainFeatures()
        
        MeanAmplitude   = extract.getMeanAmplitude(amplitudebuffer)
        MaxAmplitude    = extract.getMaxAmplitude(amplitudebuffer)
        PeaktoPeak      = extract.getPeakToPeak(amplitudebuffer)
        RMSAmplitude    = extract.getRMSAmplitude(amplitudebuffer)
        Variance        = extract.getVariance(amplitudebuffer)
        StdDev          = extract.getStdDev(amplitudebuffer)
        Skewness        = extract.getSkewness(amplitudebuffer)
        Kurtosis        = extract.getKurtosis(amplitudebuffer)
        Totalpower      = extract.gettotalpower(amplitudebuffer)
        crestfactor     = extract.getcrestfactor(amplitudebuffer)
        formfactor      = extract.getformfactor(amplitudebuffer)
        peaktomeanratio = extract.getpeaktomeanratio(amplitudebuffer)
        getmargin       = extract.getmargin(amplitudebuffer)
        RelativePekSpec = extract.getrelativepeakspectral(amplitudebuffer)
        
        fftfeatureslist = [
                    MeanAmplitude,   # 1
                    MaxAmplitude,    # 2
                    PeaktoPeak,      # 3
                    RMSAmplitude,    # 4
                    Variance,        # 5
                    StdDev,          # 6
                    Skewness,        # 7
                    Kurtosis,        # 8
                    Totalpower,      # 9
                    crestfactor,     # 10
                    formfactor,      # 11
                    peaktomeanratio, # 12
                    getmargin,       # 13
                    RelativePekSpec  # 14
                ]
        
        features_array = np.array(fftfeatureslist)
       
        features_df = pd.DataFrame(features_array, columns=frequencyspectrum)
       
        features_df.index = [
                            "MeanAmplitude", 
                            "MaxAmplitude", 
                            "PeakToPeak", 
                            "RMSAmplitude", 
                            "Variance", 
                            "StdDev", 
                            "Skewness", 
                            "Kurtosis", 
                            "TotalPower", 
                            "CrestFactor", 
                            "FormFactor", 
                            "PeakToMeanRatio", 
                            "Margin", 
                            "RelativePeakSpectral"
                        ]
        features_df = features_df.dropna() 
        # Step 1: Standardize the data (PCA works better on standardized data)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features_df.T)
        # Step 2: Apply PCA
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(scaled_data)
        return pca_result.T
         
class getFFTSignalTimeDomainFeatures:
    def __init__(self):
        self.features = {}
        
    def spectral_centroid(self,frequencyspectrum,amplitudebuffer):
        # Weighted sum of frequencies
        frequency_spectrum = np.array(frequencyspectrum)
        numerator = np.sum(frequency_spectrum * amplitudebuffer, axis=1)
        denominator = np.sum(amplitudebuffer, axis=1) + 1e-10  # Avoid division by zero
        return numerator / denominator
    
    def spectral_spread(self,frequencyspectrum,amplitudebuffer, centroid):
        centroid = np.array(centroid)
        frequency_spectrum = np.array(frequencyspectrum)
        amplitudebuffer = np.array(amplitudebuffer)
        # Spread around the centroid
        spread = np.sqrt(np.sum(((frequency_spectrum - centroid[:, None])**2) * amplitudebuffer, axis=1) /
                         (np.sum(amplitudebuffer, axis=1) + 1e-10))
        
        return spread

    def spectral_skewness(self,frequencyspectrum,amplitudebuffer, centroid, spread):
        # Skewness of frequency distribution
        centroid = np.array(centroid)
        spread = np.array(spread)
        frequency_spectrum = np.array(frequencyspectrum)
        amplitudebuffer = np.array(amplitudebuffer)
        skewness = np.sum(((frequency_spectrum - centroid[:, None])**3) * amplitudebuffer, axis=1) / \
                   (spread**3 * np.sum(amplitudebuffer, axis=1) + 1e-10)
        return skewness

    def spectral_kurtosis(self,frequencyspectrum,amplitudebuffer, centroid, spread):
        # Kurtosis of frequency distribution
        centroid = np.array(centroid)
        spread = np.array(spread)
        frequency_spectrum = np.array(frequencyspectrum)
        amplitudebuffer = np.array(amplitudebuffer)
        kurtosis = np.sum(((frequency_spectrum - centroid[:, None])**4) * amplitudebuffer, axis=1) / \
                   (spread**4 * np.sum(amplitudebuffer, axis=1) + 1e-10)
        return kurtosis

    def total_energy(self, amplitudebuffer):
        amplitudebuffer = np.array(amplitudebuffer)
        return np.sum(amplitudebuffer**2, axis=1)

    def entropy(self, amplitudebuffer):
        amplitudebuffer = np.array(amplitudebuffer)
        prob_dist = amplitudebuffer / (np.sum(amplitudebuffer, axis=1, keepdims=True) + 1e-10)
        return np.sum(entr(prob_dist), axis=1)
  
if __name__ == "__main__":
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_Me.txt'
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_110_Wall.txt'
    
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Soft/fft_Human7.txt'
    folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Hard/fft_Nothing5.txt'
    
    #folder_path = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_40_Wall.txt'
    #folder_path = 'E:/Frankfurt University of Applied Sciences/Master Thesis/GitHub/Coding/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_40_Hand.txt'
    
    fft_signal = FftSignal()
    getparameter = getFFTSignalParameters()
    extractfeatures = getFFTSignalFreqDomainFeatures()

    if folder_path is not None:
       signal_data = fft_signal.get_fft_data(folder_path)
       
       #
       # Get FFT Signal Parameters Example : Fmax, Fmin, Bandwidth Signal etc
       # #
       frequencyspectrum   = getparameter.getfrequencyspectrum(signal_data)
       amplitudebuffer     = getparameter.getamplitude(signal_data)
       absamplitudebuffer     = getparameter.getabsamplitude(signal_data)
       Fmax                = getparameter.getFmax(frequencyspectrum)
       Fmin                = getparameter.getFmin(frequencyspectrum)
       BW                  = getparameter.getBW(Fmax,Fmin)
       SamplingFrequency   = getparameter.getSamplingFrequency()
       FrequencyFactor     = getparameter.getfreqfactor(SamplingFrequency)
       FrequencyResolution = getparameter.getFreqresolution(BW)
       
       #fft_signal.plot_raw_data(frequencyspectrum,amplitudebuffer)
       MaxAmplitude    = extractfeatures.getMaxAmplitude(absamplitudebuffer)
       Totalpower      = extractfeatures.gettotalpower(absamplitudebuffer)
       
       center_frequency,F1,F2 = getparameter.getBandPassFilterParameters(frequencyspectrum,MaxAmplitude,Totalpower)
       
       smoothed_windowed_amplitudebuffer = extractfeatures.applyHanningWindow(frequencyspectrum,absamplitudebuffer,F1,F2)
       
       plt.figure(figsize=(12, 6))

       # Plot original data (without window)
       plt.subplot(1, 2, 1)
       plt.imshow(absamplitudebuffer, aspect='auto', cmap='viridis', origin='lower')
       plt.colorbar(label="Amplitude")
       plt.title("Original FFT Data (Amplitude)")
       plt.xlabel("Frequency Bin")
       plt.ylabel("Sample Index")

       # Plot windowed data (with window)
       plt.subplot(1, 2, 2)
       plt.imshow(smoothed_windowed_amplitudebuffer, aspect='auto', cmap='viridis', origin='lower')
       plt.colorbar(label="Amplitude")
       plt.title("Windowed FFT Data (After Band-Pass Filter)")
       plt.xlabel("Frequency Bin")
       plt.ylabel("Sample Index")
       plt.tight_layout()
       plt.show()
    
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # Create two subplots

       # Plot for original amplitudebuffer
       for i in range(absamplitudebuffer.shape[0]):  # Loop through each signal row
           ax1.plot(frequencyspectrum, absamplitudebuffer.iloc[i, :], alpha=0.7)
           
       ax1.set_xlabel('Frequency (Hz)')
       ax1.set_ylabel('Amplitude')
       ax1.set_title('Original Frequency Spectrum vs Amplitude Buffer')
       ax1.grid(True)
       ax1.axvline(center_frequency, color='g', linestyle='--', label=f'Center Freq: {center_frequency:.2f} Hz')
       ax1.axvline(F1, color='r', linestyle='--', label=f'Lower Cutoff: {F1:.2f} Hz')
       ax1.axvline(F2, color='r', linestyle='--', label=f'Upper Cutoff: {F2:.2f} Hz')
       ax1.legend()

        # Plot for windowed_amplitudebuffer
       for i in range(smoothed_windowed_amplitudebuffer.shape[0]):  # Loop through each signal row
           ax2.plot(frequencyspectrum, smoothed_windowed_amplitudebuffer.iloc[i, :],alpha=0.7)

       ax2.set_xlabel('Frequency (Hz)')
       ax2.set_ylabel('Amplitude')
       ax2.set_title('Windowed Frequency Spectrum vs Amplitude Buffer')
       ax2.grid(True)
       ax2.axvline(center_frequency, color='g', linestyle='--', label=f'Center Freq: {center_frequency:.2f} Hz')
       ax2.axvline(F1, color='r', linestyle='--', label=f'Lower Cutoff: {F1:.2f} Hz')
       ax2.axvline(F2, color='r', linestyle='--', label=f'Upper Cutoff: {F2:.2f} Hz')
       ax2.legend()

       plt.tight_layout()  # Adjust spacing to prevent overlap
       plt.show()    
    
       fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
       # 1st Subplot: Frequency Spectrum vs Max Amplitude
       ax1.plot(frequencyspectrum, MaxAmplitude, marker='o', linestyle='-', color='b')
       ax1.set_title('Frequency Spectrum vs Max Amplitude')
       ax1.set_xlabel('Frequency (Hz)')
       ax1.set_ylabel('Max Amplitude')
       ax1.grid(True)
       ax1.axvline(center_frequency, color='g', linestyle='--', label=f'Center Freq: {center_frequency:.2f} Hz')
       ax1.axvline(F1, color='r', linestyle='--', label=f'Lower Cutoff: {F1:.2f} Hz')
       ax1.axvline(F2, color='r', linestyle='--', label=f'Upper Cutoff: {F2:.2f} Hz')
       ax1.legend()
       
       # 2nd Subplot: Frequency Spectrum vs Total Power
       ax2.plot(frequencyspectrum, Totalpower, label="Total Power vs Frequency Spectrum", color='r')
       ax2.set_title('Frequency Spectrum vs Total Power')
       ax2.set_xlabel('Frequency Spectrum (Hz)')
       ax2.set_ylabel('Total Power')
       ax2.grid(True)
       ax2.axvline(center_frequency, color='g', linestyle='--', label=f'Center Freq: {center_frequency:.2f} Hz')
       ax2.axvline(F1, color='r', linestyle='--', label=f'Lower Cutoff: {F1:.2f} Hz')
       ax2.axvline(F2, color='r', linestyle='--', label=f'Upper Cutoff: {F2:.2f} Hz')
       ax2.legend()
     
       # 3rd Subplot: Frequency Spectrum vs Amplitude Buffer
       for i in range(absamplitudebuffer.shape[0]):  # Loop through each signal row
           ax3.plot(frequencyspectrum, absamplitudebuffer.iloc[i, :])
       ax3.set_xlabel('Frequency (Hz)')
       ax3.set_ylabel('Amplitude')
       ax3.set_title('Frequency Spectrum vs Amplitude Buffer')
       ax3.grid(True)
       ax3.axvline(center_frequency, color='g', linestyle='--', label=f'Center Freq: {center_frequency:.2f} Hz')
       ax3.axvline(F1, color='r', linestyle='--', label=f'Lower Cutoff: {F1:.2f} Hz')
       ax3.axvline(F2, color='r', linestyle='--', label=f'Upper Cutoff: {F2:.2f} Hz')
       ax3.legend()
      
       plt.tight_layout()
       plt.show()
      
       
       PCAResult = extractfeatures.getfreqPCA(frequencyspectrum,smoothed_windowed_amplitudebuffer)

       plt.figure(figsize=(10, 5))
       plt.plot(range(1, 86), PCAResult.flatten(), marker='o', linestyle='-', color='r', alpha=0.7)
       plt.xlabel("Frequency Bin Index")
       plt.ylabel("PCA Component Value")
       plt.title("PCA Across Frequency Bins (Transposed 1x85)")
       plt.grid()
       plt.show()
       print("Final PCA Result Shape:", PCAResult)
      
    #    amplitudebuffer_list = []
    #    amplitudebuffer_list.append(amplitudebuffer)
    #    amplitudebuffer_list.append(smooth)
    #    amplitudebuffer_list.append(MeanAmplitude)
    #    amplitudebuffer_list.append(MaxAmplitude)
    #    amplitudebuffer_list.append(PeaktoPeak)
    #    amplitudebuffer_list.append(RMSAmplitude)
    #    amplitudebuffer_list.append(Variance)
    #    amplitudebuffer_list.append(StdDev)
    #    amplitudebuffer_list.append(Skewness)
    #    amplitudebuffer_list.append(Kurtosis)
    #    amplitudebuffer_list.append(Totalpower)
    #    amplitudebuffer_list.append(crestfactor)
    #    amplitudebuffer_list.append(formfactor)
    #    amplitudebuffer_list.append(peaktomeanratio)
    #    amplitudebuffer_list.append(getmargin)
    #    amplitudebuffer_list.append(RelativePekSpec)
       
    #    timedomainfeatures = getFFTSignalTimeDomainFeatures()

    #    centroid = timedomainfeatures.spectral_centroid(frequencyspectrum,amplitudebuffer)
    #    spread = timedomainfeatures.spectral_spread(frequencyspectrum,amplitudebuffer, centroid)
    #    skewness = timedomainfeatures.spectral_skewness(frequencyspectrum,amplitudebuffer, centroid, spread)
    #    kurtosiss = timedomainfeatures.spectral_kurtosis(frequencyspectrum,amplitudebuffer, centroid, spread)
    #    energy = timedomainfeatures.total_energy(amplitudebuffer)
    #    entropy_values = timedomainfeatures.entropy(amplitudebuffer)

    #    rows = len(centroid)  # Should be 200
    #    x_axis = np.arange(rows)
     
    #    features = {
    #     "Spectral Centroid": centroid,
    #     "Spectral Spread": spread,
    #     "Spectral Skewness": skewness,
    #     "Spectral Kurtosis": kurtosiss,
    #     "Total Energy": energy,
    #     "Spectral Entropy": entropy_values
    #     }
       

       
       
     

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