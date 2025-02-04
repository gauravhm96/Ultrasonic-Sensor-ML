import numpy as np
import pandas as pd
import os

from tensorflow.keras import layers, models,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Cropping1D, ZeroPadding1D


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import random

import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

from FftSignalProcess import FftSignal,ExtractFeaturesFFT, getFFTSignalParameters,getFFTSignalFreqDomainFeatures

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
    
    def extractPCA(self, hard_signals, soft_signals):
        getparameter = getFFTSignalParameters()
        extractfeatures = getFFTSignalFreqDomainFeatures()
        hard_pca_results = []
        soft_pca_results = []
        for signal in hard_signals:
            frequencyspectrum   = getparameter.getfrequencyspectrum(signal)
            amplitudebuffer     = getparameter.getamplitude(signal)
            pca_result          = extractfeatures.getfreqPCA(frequencyspectrum,amplitudebuffer)
            hard_pca_results.append(pca_result)
        
        for signal in soft_signals:
            frequencyspectrum   = getparameter.getfrequencyspectrum(signal)
            amplitudebuffer     = getparameter.getamplitude(signal)
            pca_result          = extractfeatures.getfreqPCA(frequencyspectrum,amplitudebuffer)
            soft_pca_results.append(pca_result)

        return hard_pca_results,soft_pca_results
    
    def extractfftfeatureforprediction(self,signal_data):
        fft_signal = FftSignal()
        extractfeatures = statistical_features()
        timedomainfeatures = extract_time_domain_features()
        
        frequencyspectrum   = fft_signal.getfrequencyspectrum(signal_data)
        amplitudebuffer     = fft_signal.getamplitude(signal_data)
        Fmax                = fft_signal.getFmax(frequencyspectrum)
        Fmin                = fft_signal.getFmin(frequencyspectrum)
        BW                  = fft_signal.getBW(Fmax,Fmin)
        SamplingFrequency   = fft_signal.getSamplingFrequency()
        FrequencyFactor     = fft_signal.getfreqfactor(SamplingFrequency)
        FrequencyResolution = fft_signal.getFreqresolution(BW)
      
        entropy = extractfeatures.getentropy(amplitudebuffer)
        windowsize = extractfeatures.getwindowsize(entropy,FrequencyResolution)
        smooth = extractfeatures.smoothenAmplitude(amplitudebuffer,windowsize)
    
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
      
        centroid = timedomainfeatures.spectral_centroid(frequencyspectrum,amplitudebuffer)
        spread = timedomainfeatures.spectral_spread(frequencyspectrum,amplitudebuffer, centroid)
        
        timefeatures = [
                        timedomainfeatures.spectral_centroid(frequencyspectrum,amplitudebuffer),
                        timedomainfeatures.spectral_spread(frequencyspectrum,amplitudebuffer, centroid),
                        timedomainfeatures.spectral_skewness(frequencyspectrum,amplitudebuffer, centroid, spread),
                        timedomainfeatures.spectral_kurtosis(frequencyspectrum,amplitudebuffer, centroid, spread),
                        timedomainfeatures.total_energy(amplitudebuffer),
                        timedomainfeatures.entropy(amplitudebuffer)
                         ]
        
        Timefeatures_np = [feature.to_numpy() if isinstance(feature, pd.Series) else feature for feature in timefeatures]
        timefeatures_df = pd.DataFrame(Timefeatures_np).T 
               
        return freqfeatures_df,timefeatures_df

    def CnnModelFreq(self, freqfeatures_df,labels_df):
         
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
            layers.Dropout(0.5),  # Add dropout
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
         # Compile the model
         #model_freq.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        model_freq.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), 
                     loss='binary_crossentropy', metrics=['accuracy'])
        
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

    def CnnModelPCA(self, X, y):
        # Step 1: Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        
        # Step 2: Reshape the data for CNN input (samples, features, 1)
        X_train = X_train[..., np.newaxis]  # Add channel dimension
        X_test = X_test[..., np.newaxis]    # Same for the test data

        # Step 3: Dynamic batch size handling
        batch_size = min(32, X_train.shape[0])
        
        # Step 4: Build the CNN Model
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.Flatten(),
            layers.Dropout(0.5),  # Add dropout to avoid overfitting
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification (hard vs soft)
        ])
        
        # Step 5: Compile the model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                    loss='binary_crossentropy', metrics=['accuracy'])
        
        # Step 6: Print model architecture
        model.summary()
        
        # Step 7: Callbacks for training
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        
        # Step 8: Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2,
                            callbacks=[early_stopping, lr_scheduler])
        
        # Step 9: Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Model Test Accuracy: {test_accuracy:.4f}')
        
        # Step 10: Plot training history
        def plot_training_history(history):
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training History')
            plt.show()
        
        plot_training_history(history)
        
        return model, history

if __name__ == "__main__":
    
    #file = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/Machine Learning/fft_data/Hard/fft_110_Wall.txt'
    
    folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings'
    
    ffttrain = FFTModel()

    signal_data = ffttrain.load_fft_signals(folder_path)
    
    hard_signals = ffttrain.myfft_signals["Hard"]
    soft_signals = ffttrain.myfft_signals["Soft"]
    hard_pca_results, soft_pca_results = ffttrain.extractPCA(hard_signals,soft_signals)
    
    hard_pca_results = np.array(hard_pca_results)
    soft_pca_results = np.array(soft_pca_results)
    
    hard_pca_results = hard_pca_results.reshape(hard_pca_results.shape[0], -1)
    soft_pca_results = soft_pca_results.reshape(soft_pca_results.shape[0], -1)
    
    hard_labels = np.ones(len(hard_pca_results))  # Label for hard signals
    soft_labels = np.zeros(len(soft_pca_results))  # Label for soft signals

    # Step 2: Combining the Data
    # Stack the PCA results for hard and soft signals into one array (X)
    X = np.vstack((hard_pca_results, soft_pca_results))

    # Stack the labels into one array (y)
    y = np.concatenate((hard_labels, soft_labels))

    # Check the combined data and labels
    print("Combined Data (X) Shape:", X.shape)
    print("Combined Labels (y) Shape:", y.shape)
    
    model, history = ffttrain.CnnModelPCA(X, y)
    
    model.save("C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/pca_cnn_model.h5")
    
    #output_folder = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Test/Soft'
    
    # for idx, pca_result in enumerate(hard_pca_results):
    #     # Create the plot
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(range(1, 86), pca_result.flatten(), marker='o', linestyle='-', color='r', alpha=0.7)
    #     plt.xlabel("Frequency Bin Index")
    #     plt.ylabel("PCA Component Value")
    #     plt.title(f"PCA Across Frequency Bins {idx+1} (Transposed 1x85)")
    #     plt.grid()

    #     # Save the plot as a PNG file
    #     plot_filename = f"PCA_plot_{idx+1}.png"
    #     plot_path = os.path.join(output_folder, plot_filename)
    #     plt.savefig(plot_path)

    #     # Close the plot to avoid memory issues
    #     plt.close()

    #     print(f"Saved plot for PCA Result {idx+1} at {plot_path}")
    
    # for idx, pca_result in enumerate(soft_pca_results):
    #      # Create the plot
    #      plt.figure(figsize=(10, 5))
    #      plt.plot(range(1, 86), pca_result.flatten(), marker='o', linestyle='-', color='r', alpha=0.7)
    #      plt.xlabel("Frequency Bin Index")
    #      plt.ylabel("PCA Component Value")
    #      plt.title(f"PCA Across Frequency Bins {idx+1} (Transposed 1x85)")
    #      plt.grid()

    #      # Save the plot as a PNG file
    #      plot_filename = f"PCA_plot_{idx+1}.png"
    #      plot_path = os.path.join(output_folder, plot_filename)
    #      plt.savefig(plot_path)

    #      # Close the plot to avoid memory issues
    #      plt.close()

    #     print(f"Saved plot for PCA Result {idx+1} at {plot_path}")

    # print("___________________________________________________________________________________")
    # print("___________________________________________________________________________________")

    
    # # freqfeatures_df,timefeatures_df = ffttrain.extractfftfeatures(signal_data)
    # freqfeatures_df,labels_df = ffttrain.extractfftfeatures(hard_signals,soft_signals)
    
    #signal = ffttrain.load_data(file)

    #model_freq, history_freq = ffttrain.CnnModelFreq(aggregated_freqfeatures,labels)
    
    
    # file_path = r"C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Test/output2.txt"  # Change to your desired path
    # ffttrain.save_signal_data(freqfeatures_df, timefeatures_df, labels_df, file_path)
    
    
    #model_freq.save("C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/my_model.h5")
    
    #ffttrain.CnnModelTime(timefeatures_df,labeltime)
    
    # loaded_model = load_model("C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/my_model.h5")
    
    # predictfile = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Soft/fft_Human5.txt'
    # #predictfile = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Hard/fft_Nothing3.txt'
    
    # signal_data = ffttrain.load_data(predictfile)
    # freqfeatures_df,timefeatures_df = ffttrain.extractfftfeatureforprediction(signal_data)
    # print(freqfeatures_df)
    # scaler = StandardScaler()
    # freqfeatures_df_scaled = scaler.fit_transform(freqfeatures_df)  
    # freqfeatures_df = pd.DataFrame(freqfeatures_df_scaled, columns=freqfeatures_df.columns)
    # min_samples = len(freqfeatures_df)
    # freqfeatures_df = freqfeatures_df.iloc[:min_samples, :]
    
    # freqfeatures_array = freqfeatures_df.to_numpy()
    # freqfeatures_array = freqfeatures_array[..., np.newaxis]
    # # Check the shape of X_freq
    # print("Shape of X_freq:", freqfeatures_array.shape)

    
    # predictions = loaded_model.predict(freqfeatures_array)
    # predicted_labels = predictions.flatten()
    # count_zeros = np.sum(predicted_labels == 0)
    # count_ones = np.sum(predicted_labels == 1)
    
    # if count_zeros > count_ones:
    #     majority_label = 0
    # else:
    #     majority_label = 1

    # print(f"Predicted Majority Label: {majority_label}")
    

    
    