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

class PredictFFT:
    def __init__(self):
        pass
    
    def loadCNNModel(self,data_dir):
        model = load_model(data_dir)
        return model
    
    def loadpredictfile(self,file):
        fft_signal = FftSignal()
        getparameter = getFFTSignalParameters()
        extractfeatures = getFFTSignalFreqDomainFeatures()
        if file is not None:
           signal_data = fft_signal.get_fft_data(file)
           
           frequencyspectrum   = getparameter.getfrequencyspectrum(signal_data)
           amplitudebuffer     = getparameter.getamplitude(signal_data)
           PCAResult = extractfeatures.getfreqPCA(frequencyspectrum,amplitudebuffer)
           
           plt.figure(figsize=(10, 5))
           plt.plot(range(1, 86), PCAResult.flatten(), marker='o', linestyle='-', color='r', alpha=0.7)
           plt.xlabel("Frequency Bin Index")
           plt.ylabel("PCA Component Value")
           plt.title("PCA Across Frequency Bins (Transposed 1x85)")
           plt.grid()
           plt.show()
           print("Final PCA Result Shape:", PCAResult)
        
        return PCAResult

if __name__ == "__main__":
    
    folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Soft/fft_Human4.txt'
    #folder_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Hard/fft_Nothing10.txt'
    
    Cnnmodel_path = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/pca_cnn_model.h5'
    
    Predict = PredictFFT()
    PredictPCA  = Predict.loadpredictfile(folder_path)
    
    UpdatedPredictPCA = PredictPCA.reshape(PredictPCA.shape[0], -1)
    
    model = Predict.loadCNNModel(Cnnmodel_path)
    
    predictions = model.predict(UpdatedPredictPCA)
    
    predicted_labels = (predictions > 0.5).astype(int)
    
    for label in predicted_labels:
        if label == 0:
            print("Soft")
        else:
            print("Hard")