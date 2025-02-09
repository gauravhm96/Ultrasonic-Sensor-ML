import os

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from FftSignalProcess import (
    FftSignal,
    getFFTSignalFreqDomainFeatures,
    getFFTSignalParameters,
)


class Predict_FFT:
    def __init__(self):
        pass

    def loadCNNModel(self, data_dir):
        model = load_model(data_dir)
        return model

    def loadpredictfile(self, file):
        fft_signal = FftSignal()
        getparameter = getFFTSignalParameters()
        extractfeatures = getFFTSignalFreqDomainFeatures()
        if file is not None:
            signal_data = fft_signal.get_fft_data(file)

            frequencyspectrum = getparameter.getfrequencyspectrum(signal_data)
            amplitudebuffer = getparameter.getabsamplitude(signal_data)
            
            MaxAmplitude    = extractfeatures.getMaxAmplitude(amplitudebuffer)
            Totalpower      = extractfeatures.gettotalpower(amplitudebuffer)
            center_frequency,F1,F2 = getparameter.getBandPassFilterParameters(frequencyspectrum,MaxAmplitude,Totalpower)
            amplitudebuffer = extractfeatures.applyHanningWindow(frequencyspectrum,amplitudebuffer,F1,F2)
            
            pca_result = extractfeatures.getfreqPCA(frequencyspectrum, amplitudebuffer)
        return pca_result


if __name__ == "__main__":

    FOLDER_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Soft/fft_Human4.txt"
    #FOLDER_PATH = 'C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/fft_data/New Readings/Hard/fft_Nothing10.txt'

    CNNMODEL_PATH = "C:/@DevDocs/Projects/Mine/New folder/Ultrasonic-Sensor-ML/UltrasonicSensorApp/Model/pca_cnn_model.h5"

    Predict = Predict_FFT()
    PredictPCA = Predict.loadpredictfile(FOLDER_PATH)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 86),PredictPCA.flatten(),marker="o",linestyle="-",color="r",alpha=0.7,)
    plt.xlabel("Frequency Bin Index")
    plt.ylabel("PCA Component Value")
    plt.title("PCA Across Frequency Bins (Transposed 1x85)")
    plt.grid()
    plt.show()
    print("Final PCA Result Shape:", PredictPCA)

    UpdatedPredictPCA = PredictPCA.reshape(PredictPCA.shape[0], -1)

    model = Predict.loadCNNModel(CNNMODEL_PATH)

    predictions = model.predict(UpdatedPredictPCA)

    predicted_labels = (predictions > 0.5).astype(int)

    for label in predicted_labels:
        if label == 0:
            print("Soft")
        else:
            print("Hard")
