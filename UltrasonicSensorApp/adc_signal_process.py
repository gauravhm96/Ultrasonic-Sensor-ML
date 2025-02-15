import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, peak_prominences, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split


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

    def plot_signal_with_peaks(self, dataframe, peak_index, peak_value):
        plt.figure(figsize=(15, 8))
        for index, row in dataframe.iterrows():
            plt.plot(
                row, color="lightgray", linewidth=0.5, alpha=0.7
            )  # Plotting each signal

        # Highlight the most prominent peak on all signals using lines
        plt.axvline(x=peak_index, color="red", linestyle="--", label="Peak Index")
        plt.axhline(y=peak_value, color="blue", linestyle="--", label="Peak Amplitude")

        plt.title("All Filtered ADC Signals with Highlighted Peak")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class ADCSignalProcess:
    def __init__(self):
        pass

    def performfft(self, data, samplingfreq):
        dt = 1 / samplingfreq
        data = data.to_numpy()
        N = data.shape[1]

        # Compute the FFT for each measurement (each row) using real FFT (rfft)
        fft_results = np.apply_along_axis(np.fft.rfft, 1, data)
        fft_magnitude = np.abs(fft_results)

        avg_fft_magnitude = np.mean(fft_magnitude, axis=0)

        # Calculate the frequency bins for the FFT
        freqs = np.fft.rfftfreq(N, d=dt)

        dynamic_multiplier = 1.2  # Adjust this multiplier as needed
        dynamic_threshold = np.mean(avg_fft_magnitude) + dynamic_multiplier * np.std(
            avg_fft_magnitude
        )

        indices_above_threshold = np.where(avg_fft_magnitude > dynamic_threshold)[0]

        if indices_above_threshold.size > 0:
            start_index = indices_above_threshold[0]
            end_index = indices_above_threshold[-1]

            # Convert indices to frequency values using the frequency bins
            f_start = freqs[start_index]
            f_end = freqs[end_index]

            print("Start frequency: {:.2f} Hz".format(f_start))
            print("End frequency: {:.2f} Hz".format(f_end))
        else:
            print("No frequency components exceed the threshold.")

        # plt.figure(figsize=(12, 6))
        # plt.plot(freqs, avg_fft_magnitude, label="Avg FFT Magnitude", color="blue")
        # plt.axhline(
        #     y=dynamic_threshold,
        #     color="green",
        #     linestyle="--",
        #     label=f"Threshold ({dynamic_threshold*100:.0f}% of max)",
        # )
        # plt.axvline(
        #     x=f_start,
        #     color="red",
        #     linestyle="--",
        #     label=f"Start Frequency: {f_start:.0f} Hz",
        # )
        # plt.axvline(
        #     x=f_end,
        #     color="magenta",
        #     linestyle="--",
        #     label=f"End Frequency: {f_end:.0f} Hz",
        # )
        # plt.xlim(30000, 50000)
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Magnitude")
        # plt.title("Averaged FFT Magnitude with Detected Signal Band")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        
        return freqs,avg_fft_magnitude,dynamic_threshold,f_start,f_end

    # Designs a Butterworth band-pass filter.
    # Parameters:
    # - lowcut: Lower cutoff frequency in Hz.
    # - highcut: Upper cutoff frequency in Hz.
    # - fs: Sampling frequency in Hz.
    # - order: Order of the filter (default is 5).
    # Returns:
    # - sos: Second-order sections for the filter.
    def butter_bandpass(self, lowcut, highcut, fs):
        order = 5
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], analog=False, btype="band", output="sos")
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
    def apply_bandpass_filter(self, data, sos):
        filtered_data = sosfiltfilt(sos, data, axis=-1)
        return filtered_data

    def detect_prominent_peaks(self, dataframe, std_multiplier=3):
        max_signal = dataframe.max(axis=0).to_numpy()

        # Set prominence threshold dynamically
        prominence_threshold = np.mean(max_signal) + std_multiplier * (
            np.std(max_signal)
        )

        # Find peaks in the mean signal with the specified prominence
        peaks, properties = find_peaks(max_signal, prominence=prominence_threshold)

        if len(peaks) > 0:
            prominent_peak_index = peaks[0]  # Take the first prominent peak
            prominent_peak_value = max_signal[prominent_peak_index]
            return prominent_peak_index, prominent_peak_value
        return None, None

    def calculate_distance_from_peak(self, highest_peak_index):
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

    def detect_prominent_peaks(self, dataframe, std_multiplier=3):
        results = []
        for idx, row in dataframe.iterrows():
            signal = row.to_numpy()
            candidate_peaks, _ = find_peaks(signal)
            prominences, left_bases, right_bases = peak_prominences(
                signal, candidate_peaks
            )
            abs_signal = np.abs(signal)
            dynamic_threshold = np.mean(abs_signal) + std_multiplier * np.std(
                abs_signal
            )

            labels = []
            for prom in prominences:
                if prom >= dynamic_threshold:
                    labels.append("peak")
                else:
                    labels.append("non-peak")

            # Retrieve amplitudes of candidate peaks.
            amplitudes = signal[candidate_peaks]

            # Save the information in a dictionary.
            results.append(
                {
                    "measurement_index": idx,
                    "candidate_peaks": candidate_peaks,
                    "amplitudes": amplitudes,
                    "prominences": prominences,
                    "labels": labels,
                }
            )

        return results

    def consolidate_peak_features(self, peak_features):
        rows = []
        for measurement in peak_features:
            meas_index = measurement["measurement_index"]
            candidate_peaks = measurement["candidate_peaks"]
            amplitudes = measurement["amplitudes"]
            prominences = measurement["prominences"]
            labels = measurement["labels"]

            for i in range(len(candidate_peaks)):
                row = {
                    "measurement_index": meas_index,
                    "candidate_peak_index": candidate_peaks[i],
                    "amplitude": amplitudes[i],
                    "prominence": prominences[i],
                    "label": 1 if labels[i] == "peak" else 0,
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    # def peaks_model(self, df_features, FILE_PATH):
    #     X = df_features[["candidate_peak_index", "amplitude", "prominence"]]
    #     y = df_features["label"]
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.3, random_state=42
    #     )

    #     # Train a simple logistic regression model
    #     model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
    #     model.fit(X_train, y_train)

    #     # Predict on the test set
    #     y_pred = model.predict(X_test)
    #     print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    #     print("\nClassification Report:")
    #     print(classification_report(y_test, y_pred))

    #     # Set up 5-fold cross-validation
    #     kf = KFold(n_splits=5, shuffle=True, random_state=42)

    #     # Perform cross-validation using accuracy as the metric
    #     cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

    #     # Print the cross-validation scores and the mean accuracy
    #     print("Cross-Validation Accuracy Scores:", cv_scores)
    #     print("Mean CV Accuracy:", np.mean(cv_scores))

    #     joblib.dump(model, FILE_PATH)
    
    def peaks_model(self, df_features):
        feature_columns = ['candidate_peak_index', 'amplitude', 'prominence', 'distance']
        X = df_features[feature_columns]
        y = df_features["label"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a simple logistic regression model
        model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Set up 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validation using accuracy as the metric
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

        # Print the cross-validation scores and the mean accuracy
        print("Cross-Validation Accuracy Scores:", cv_scores)
        print("Mean CV Accuracy:", np.mean(cv_scores))
        return model
    

    def process_adc_data_dataframe(self, dataframe, std_multiplier=3):
        return self.detect_prominent_peaks(dataframe, std_multiplier)

class PredictADC:
    def identify_first_echo(self, df_features_with_predictions):
        first_echo_list = []
        # Get the unique measurement indices
        unique_measurements = df_features_with_predictions["measurement_index"].unique()

        # Process each measurement individually
        for meas in unique_measurements:
            # Filter rows for the current measurement
            df_meas = df_features_with_predictions[
                df_features_with_predictions["measurement_index"] == meas
            ]
            # Filter only those candidate peaks predicted as significant (label == 1)
            df_significant = df_meas[df_meas["predicted_label"] == 1]

            if not df_significant.empty:
                # Select the candidate with the smallest candidate_peak_index (i.e., earliest in time)
                first_echo = df_significant.loc[
                    df_significant["candidate_peak_index"].idxmin()
                ]
                first_echo_list.append(
                    {
                        "measurement_index": meas,
                        "first_echo_candidate_peak_index": first_echo[
                            "candidate_peak_index"
                        ],
                        "amplitude": first_echo["amplitude"],
                        "prominence": first_echo["prominence"],
                    }
                )
            else:
                # If no candidate was predicted as a significant peak, record None values
                first_echo_list.append(
                    {
                        "measurement_index": meas,
                        "first_echo_candidate_peak_index": None,
                        "amplitude": None,
                        "prominence": None,
                    }
                )

        # Return a DataFrame summarizing the first echo for each measurement
        return pd.DataFrame(first_echo_list)


if __name__ == "__main__":
    # folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Soft/fft_Me.txt'

    FILE_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Raw_Data/adc_100.txt"
    FOLDER_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Raw_Data/"
    
    OUTPUT_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Test/adc_120.txt"

    myadcdata = ADCSignal()
    processadcdata = ADCSignalProcess()
    myplot = Plot_signals()
    train = TrainADC()
    predictdata = PredictADC()
    
    lowcut = 39500.0  # Lower cutoff frequency in Hz
    highcut = 41500.0  # Upper cutoff frequency in Hz
    fs = 1.953125e6  # Sampling frequency in Hz (as per RED Pitaya)
    sos = processadcdata.butter_bandpass(lowcut, highcut, fs)

    my_adc_data = myadcdata.get_adc_data(FILE_PATH)
    
    adc_data_array = my_adc_data.to_numpy()
    filtered_data_array = np.apply_along_axis(
         processadcdata.apply_bandpass_filter, 1, adc_data_array, sos
     )
    filtered_myadc_data = pd.DataFrame(filtered_data_array)

    processadcdata.performfft(filtered_myadc_data, fs)
    
    peak_index, peak_value = processadcdata.detect_prominent_peaks(filtered_myadc_data)

    print("Highest Peak Index:", peak_index)
    print("Highest Peak Amplitude:", peak_value)
    
    print("Object Distance:", processadcdata.calculate_distance_from_peak(peak_index))
    
    peak_features = train.process_adc_data_dataframe(filtered_myadc_data)
    df_features = train.consolidate_peak_features(peak_features)
    
    # MODEL_FILE_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/peak_classifier_model1.pkl"

    # train.peaks_model(df_features, MODEL_FILE_PATH)
    
    # loaded_model = joblib.load(MODEL_FILE_PATH)

    # df_features = df_features.copy()
    # feature_columns = ["candidate_peak_index", "amplitude", "prominence"]
    # df_features["predicted_label"] = loaded_model.predict(df_features[feature_columns])

    # df_first_echo = predictdata.identify_first_echo(df_features)

    # predicted_peaks = df_features[df_features["predicted_label"] == 1].copy()

    # # Check if any predicted peaks exist
    # if not predicted_peaks.empty:
    #     # Find the row with the maximum amplitude among predicted peaks
    #     max_peak_row = predicted_peaks.loc[predicted_peaks["amplitude"].idxmax()]

    #     highest_peak_index = max_peak_row["candidate_peak_index"]
    #     highest_peak_amplitude = max_peak_row["amplitude"]

    #     print("Highest Predicted Peak Index:", highest_peak_index)
    #     print("Highest Predicted Peak Amplitude:", highest_peak_amplitude)
    # else:
    #     print("No predicted peaks were found.")


    
    adc_data_list = []
    
    # Iterate over all files in the folder, label index starts from 1
    for idx, filename in enumerate(os.listdir(FOLDER_PATH), start=1):
        # Optionally, only process .txt files (adjust extension if needed)
        if filename.endswith('.txt'):
            FILE_PATH = os.path.join(FOLDER_PATH, filename)
            # Read ADC data from file using your get_adc_data function
            my_adc_data = myadcdata.get_adc_data(FILE_PATH)
            # Append a tuple containing the label (index) and the ADC data
            adc_data_list.append((idx, my_adc_data))

    filtered_data_list = []
    
    for label, my_adc_data in adc_data_list:
        # Convert the DataFrame to a NumPy array
        adc_data_array = my_adc_data.to_numpy()
        
        # Apply the bandpass filter to each row of the ADC data array
        filtered_data_array = np.apply_along_axis(processadcdata.apply_bandpass_filter, 1, adc_data_array, sos)
        
        # Convert the filtered data back to a DataFrame
        filtered_myadc_data = pd.DataFrame(filtered_data_array)
        
        # Append the tuple (label, filtered ADC data) to the list
        filtered_data_list.append((label, filtered_myadc_data))
        
    peak_features = train.process_adc_data_dataframe(filtered_myadc_data)
    df_features = train.consolidate_peak_features(peak_features)
    
    distance_to_features = {}
    
    for file_label, filtered_df in filtered_data_list:
        # Step 1: Process the filtered ADC DataFrame to extract candidate peak features.
        peak_features = train.process_adc_data_dataframe(filtered_df)
        
        # Step 2: Consolidate these features into a DataFrame.
        df_features = train.consolidate_peak_features(peak_features)
        
        # Step 3: Detect the prominent peak from the filtered data.
        peak_index, peak_value = processadcdata.detect_prominent_peaks(filtered_df)
        
        # Step 4: Calculate the distance corresponding to the detected peak index.
        distance = processadcdata.calculate_distance_from_peak(peak_index)
        
        # Step 5: Store the consolidated features in a dictionary, keyed by the calculated distance.
        distance_to_features[distance] = df_features
        
    dfs = []
    for dist, df in distance_to_features.items():
        df_copy = df.copy()  # avoid modifying the original DataFrame
        df_copy['distance'] = dist  # add the distance as a new column
        dfs.append(df_copy)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    MODEL_FILE_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/peak_classifier_model2.pkl"
    
    trained_model = train.peaks_model(combined_df)
    
    joblib.dump(trained_model, MODEL_FILE_PATH)
    
    loaded_model = joblib.load(MODEL_FILE_PATH)

    df_features = df_features.copy()
    df_features["distance"] = 0.0
    feature_columns = ['candidate_peak_index', 'amplitude', 'prominence', 'distance']   
    df_features["predicted_label"] = loaded_model.predict(df_features[feature_columns])

    df_first_echo = predictdata.identify_first_echo(df_features)

    predicted_peaks = df_features[df_features["predicted_label"] == 1].copy()

    # Check if any predicted peaks exist
    if not predicted_peaks.empty:
        # Find the row with the maximum amplitude among predicted peaks
        max_peak_row = predicted_peaks.loc[predicted_peaks["amplitude"].idxmax()]

        highest_peak_index = max_peak_row["candidate_peak_index"]
        highest_peak_amplitude = max_peak_row["amplitude"]

        print("Highest Predicted Peak Index:", highest_peak_index)
        print("Highest Predicted Peak Amplitude:", highest_peak_amplitude)
    else:
        print("No predicted peaks were found.")


    