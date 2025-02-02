import numpy as np
import pandas as pd
import tensorflow as tf
import os

from tensorflow.keras import layers, models,Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import joblib

from FftSignalProcess import FftSignal,statistical_features, extract_time_domain_features

class FFTModel:
    def __init__(self):
        self.myfft_signals = {
                            "Hard": [],
                            "Soft": []
                           }
        
    def load_fft_signals(self,data_dir):
        fft_signal = FftSignal()
        for category in ["Hard", "Soft"]:
            folder_path = os.path.join(data_dir, category)
            if os.path.exists(folder_path):  # Check if folder exists
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    try:
                        signal_data = fft_signal.get_fft_data(file_path)
                        self.myfft_signals[category].append(signal_data) 
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    def load_data(self,file_path):
        fft_signal = FftSignal()
        signal_data = fft_signal.get_fft_data(file_path)
        return signal_data
    
    # def extractfftfeatures(self,signal_data):
    #     fft_signal = FftSignal()
    #     extractfeatures = statistical_features()
    #     timedomainfeatures = extract_time_domain_features()
        
    #     frequencyspectrum   = fft_signal.getfrequencyspectrum(signal_data)
    #     amplitudebuffer     = fft_signal.getamplitude(signal_data)
    #     Fmax                = fft_signal.getFmax(frequencyspectrum)
    #     Fmin                = fft_signal.getFmin(frequencyspectrum)
    #     BW                  = fft_signal.getBW(Fmax,Fmin)
    #     SamplingFrequency   = fft_signal.getSamplingFrequency()
    #     FrequencyFactor     = fft_signal.getfreqfactor(SamplingFrequency)
    #     FrequencyResolution = fft_signal.getFreqresolution(BW)
        
    #     entropy = extractfeatures.getentropy(amplitudebuffer)
    #     windowsize = extractfeatures.getwindowsize(entropy,FrequencyResolution)
    #     smooth = extractfeatures.smoothenAmplitude(amplitudebuffer,windowsize)
    
    #     freqfeatures = [
    #                 extractfeatures.getMeanAmplitude(amplitudebuffer),
    #                 extractfeatures.getMaxAmplitude(amplitudebuffer),
    #                 extractfeatures.getPeakToPeak(amplitudebuffer),
    #                 extractfeatures.getRMSAmplitude(amplitudebuffer),
    #                 extractfeatures.getVariance(amplitudebuffer),
    #                 extractfeatures.getStdDev(amplitudebuffer),
    #                 extractfeatures.getSkewness(amplitudebuffer),
    #                 extractfeatures.getKurtosis(amplitudebuffer),
    #                 extractfeatures.gettotalpower(amplitudebuffer),
    #                 extractfeatures.getcrestfactor(amplitudebuffer),
    #                 extractfeatures.getformfactor(amplitudebuffer),
    #                 extractfeatures.getpeaktomeanratio(amplitudebuffer),
    #                 extractfeatures.getmargin(amplitudebuffer),
    #                 extractfeatures.getrelativepeakspectral(amplitudebuffer)
    #             ]
        
    #     freqfeatures_df = pd.DataFrame(freqfeatures).T
        
    #     centroid = timedomainfeatures.spectral_centroid(frequencyspectrum,amplitudebuffer)
    #     spread = timedomainfeatures.spectral_spread(frequencyspectrum,amplitudebuffer, centroid)
        
    #     timefeatures = [
    #                     timedomainfeatures.spectral_centroid(frequencyspectrum,amplitudebuffer),
    #                     timedomainfeatures.spectral_spread(frequencyspectrum,amplitudebuffer, centroid),
    #                     timedomainfeatures.spectral_skewness(frequencyspectrum,amplitudebuffer, centroid, spread),
    #                     timedomainfeatures.spectral_kurtosis(frequencyspectrum,amplitudebuffer, centroid, spread),
    #                     timedomainfeatures.total_energy(amplitudebuffer),
    #                     timedomainfeatures.entropy(amplitudebuffer)
    #                     ]
        
    #     Timefeatures_np = [feature.to_numpy() if isinstance(feature, pd.Series) else feature for feature in timefeatures]
    #     timefeatures_df = pd.DataFrame(Timefeatures_np).T 
                
    #     return freqfeatures_df,timefeatures_df

    def extractfftfeatures(self, hard_signals, soft_signals):
        fft_signal = FftSignal()
        extractfeatures = statistical_features()

        freqfeatures_list = []
        labels = []

        # Process both categories
        for category, signals in [("Hard", hard_signals), ("Soft", soft_signals)]:
            for signal_data in signals:
                frequencyspectrum = fft_signal.getfrequencyspectrum(signal_data)
                amplitudebuffer = fft_signal.getamplitude(signal_data)
                Fmax = fft_signal.getFmax(frequencyspectrum)
                Fmin = fft_signal.getFmin(frequencyspectrum)
                BW = fft_signal.getBW(Fmax, Fmin)
                SamplingFrequency = fft_signal.getSamplingFrequency()
                FrequencyFactor = fft_signal.getfreqfactor(SamplingFrequency)
                FrequencyResolution = fft_signal.getFreqresolution(BW)

                entropy = extractfeatures.getentropy(amplitudebuffer)
                windowsize = extractfeatures.getwindowsize(entropy, FrequencyResolution)
                smooth = extractfeatures.smoothenAmplitude(amplitudebuffer, windowsize)

                # Extract frequency features
                freqfeatures = [
                    extractfeatures.getMeanAmplitude(amplitudebuffer),
                    extractfeatures.getMaxAmplitude(amplitudebuffer),
                    extractfeatures.getPeakToPeak(amplitudebuffer),
                    extractfeatures.getRMSAmplitude(amplitudebuffer),
                    extractfeatures.getVariance(amplitudebuffer),
                    extractfeatures.getStdDev(amplitudebuffer),
                    extractfeatures.getSkewness(amplitudebuffer),
                    extractfeatures.getKurtosis(amplitudebuffer),
                    extractfeatures.gettotalpower(amplitudebuffer),
                    extractfeatures.getcrestfactor(amplitudebuffer),
                    extractfeatures.getformfactor(amplitudebuffer),
                    extractfeatures.getpeaktomeanratio(amplitudebuffer),
                    extractfeatures.getmargin(amplitudebuffer),
                    extractfeatures.getrelativepeakspectral(amplitudebuffer)
                ]
                
                freqfeatures_df = pd.DataFrame(freqfeatures).T
                freqfeatures_list.append(freqfeatures_df)
                
                num_rows_per_signal = freqfeatures_df.shape[0]
                # Create the corresponding labels for the signal
                label_value = 1 if category == "Hard" else 0
                signal_labels = [label_value] * num_rows_per_signal  # Repeat label for each row in the signal
                labels.extend(signal_labels)


        freqfeatures_df = pd.concat(freqfeatures_list, ignore_index=True)
        labels_df = pd.DataFrame(labels, columns=["label"])
    
        return freqfeatures_df, labels_df
    
    def CnnModelFreq(self, freqfeatures_df,labels_df):
        # Split the data into training and testing
        min_samples = len(freqfeatures_df)
        freqfeatures_df = freqfeatures_df.iloc[:min_samples, :]
        labels_df = labels_df[:min_samples]
        
        # Split into train and test sets
        X_train_freq, X_test_freq, y_train, y_test = train_test_split(freqfeatures_df, labels_df, test_size=0.2, random_state=42)
        
        print("X_train_freq shape:", X_train_freq.shape)
        print("y_train shape:", y_train.shape)
        
         # Convert the labels (y_train and y_test) to NumPy arrays
        X_train_freq = np.array(X_train_freq)
        X_test_freq = np.array(X_test_freq)
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        
        
        # Reshape the data to be 2D (samples, features, 1) as CNNs expect 3D inputs
        X_train_freq = X_train_freq[..., np.newaxis]  # Use .values to convert DataFrame to ndarray, then add channel dimension
        X_test_freq = X_test_freq[..., np.newaxis]    # Same for the test data
        
        # Dynamic batch size handling
        batch_size = min(32, X_train_freq.shape[0])
        
        # CNN Model for freqfeatures_df (Model 1)
        model_freq = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=(X_train_freq.shape[1], X_train_freq.shape[2])),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        # Compile the model
        model_freq.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Print model architecture
        model_freq.summary()
        
        # Callbacks for better training
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        # Train the model
        history_freq = model_freq.fit(X_train_freq, y_train, epochs=50, batch_size=batch_size, validation_split=0.2,
                                  callbacks=[early_stopping, lr_scheduler])

        # Evaluate the model
        test_loss_freq, test_accuracy_freq = model_freq.evaluate(X_test_freq, y_test)
        print(f'Frequency Model Test Accuracy: {test_accuracy_freq:.4f}')
        
        # Plot training history
        def plot_training_history(history):
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training History')
            plt.show()
            
        plot_training_history(history_freq)
            
        return model_freq, history_freq
    
    def CnnModelTime(self, timefeatures_df, label):
        # Split the data into training and testing
        min_samples = len(timefeatures_df)
        timefeatures_df = timefeatures_df.iloc[:min_samples, :]
        label = label[:min_samples]

        # Split into train and test sets
        X_train_time, X_test_time, y_train, y_test = train_test_split(timefeatures_df, label, test_size=0.2, random_state=42)
        
        # Reshape the data to be 2D (samples, features, 1) as CNNs expect 3D inputs
        X_train_time = X_train_time.values[..., np.newaxis]  # Add channel dimension
        X_test_time = X_test_time.values[..., np.newaxis]    # Same for the test data
        
        # CNN Model for timefeatures_df (Model 1)
        model_time = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=(X_train_time.shape[1], X_test_time.shape[2])),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        # Compile the model
        model_time.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history_time = model_time.fit(X_train_time, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the model
        test_loss_time, test_accuracy_time = model_time.evaluate(X_test_time, y_test)
        print(f'Time Model Test Accuracy: {test_accuracy_time:.4f}')
        
        return model_time, history_time
    

    def load_target_labels(self, file_path, num_samples):
        if "Hard" in file_path:
            label = np.ones(num_samples)  # Label 1 for Hard
        elif "Soft" in file_path:
            label = np.zeros(num_samples)  # Label 0 for Soft
        else:
            raise ValueError("File path must contain 'Hard' or 'Soft' to assign a label.")
        return label
    
if __name__ == "__main__":
    
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_110_Wall.txt'
    
    folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data'
    
    ffttrain = FFTModel()
    signal_data = ffttrain.load_fft_signals(folder_path)
    
    hard_signals = ffttrain.myfft_signals["Hard"]
    soft_signals = ffttrain.myfft_signals["Soft"]
    
    # freqfeatures_df,timefeatures_df = ffttrain.extractfftfeatures(signal_data)
    freqfeatures_df,labels_df = ffttrain.extractfftfeatures(hard_signals,soft_signals)
    
    # file_path = r"C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Test/output2.txt"  # Change to your desired path
    # ffttrain.save_signal_data(freqfeatures_df, timefeatures_df, labels_df, file_path)
    
    ffttrain.CnnModelFreq(freqfeatures_df,labels_df)
    
    # labeltime = ffttrain.load_target_labels(folder_path, timefeatures_df.shape[0])
    #ffttrain.CnnModelTime(timefeatures_df,labeltime)
    
    
    
   