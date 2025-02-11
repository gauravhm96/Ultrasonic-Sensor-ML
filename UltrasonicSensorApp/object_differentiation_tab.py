from PyQt5.QtWidgets import QLabel, QHBoxLayout, QPushButton, QFileDialog,QSizePolicy,QDialog,QSpacerItem,QProgressBar
from FftSignalProcess import FftSignal,getFFTSignalParameters,getFFTSignalFreqDomainFeatures
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtWidgets import QDialog, QVBoxLayout
import pandas as pd
from PyQt5.QtCore import Qt
import numpy as np

from trainfftsignal import FFTModel
from Predictfft import Predict_FFT
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap

signal_data = None
file_path = None
folder_path = None
model = None
history = None
predictmodelpath = None
predictfilepath = None
Loadcnnmodel = None

def object_differentiation_features(layout, output_box):
    
    fft_signal = FftSignal()
    getparameter = getFFTSignalParameters()
    extract = getFFTSignalFreqDomainFeatures()
    feature_label = QLabel("Computation for FFT Data Acquired from Ultrasonic Sensor for Object Differentiation")
    feature_label.setStyleSheet("font-size: 18px; font-weight: normal;padding: 5px")
    layout.addWidget(feature_label)

    file_layout = QHBoxLayout()
    file_layout.setSpacing(10)      
    file_layout.setContentsMargins(0, 0, 0, 0)      
    file_layout.setAlignment(Qt.AlignLeft)
    
    select_file_button = QPushButton("Select FFT Signal")
    select_file_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    select_file_button.setFixedWidth(200)
    select_file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    # Add button to layout
    file_layout.addWidget(select_file_button)

    # Add a QLabel to show the file path
    file_path_label = QLabel("No file selected")
    file_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    file_layout.addWidget(file_path_label)

    layout.addLayout(file_layout)
    # Function to open file dialog and set path
    def select_file():
        global signal_data
        global file_path
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Select Signal File", "", "fft signal (*.txt);;CSV (*.csv)"
        )
        if file_path:
            if "fft" not in file_path.lower():
                output_box.append("I think you have not selected Fft Data :( ")
                return
            else :
                file_path_label.setText(file_path)
                output_box.append(f"File selected: {file_path}")
                try:
                    signal_data = fft_signal.get_fft_data(file_path)
                    output_box.append("FFT Data Loaded Successfully.\n"
                                  f"Total Data Points: {len(signal_data)}\n"
                                  "Processing data...\n")
                    output_box.append(f"Signal:\n{signal_data[:10]}")
                except Exception as e:
                    output_box.append(f"Error loading FFT data: {e}")

    # Connect the button to the select file function
    select_file_button.clicked.connect(select_file)
    
    # Create a horizontal layout for all buttons
    button_layout = QHBoxLayout()
    button_layout.setSpacing(10)  # Remove space between buttons
    button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
    button_layout.setAlignment(Qt.AlignLeft)
    
    # Add the required buttons
    buttons = ["Save Data", "Show Heat Map", "Signal Characetristics", "Feature Extract(Freq)","Generate PCA"]
    for button_name in buttons:
        button = QPushButton(button_name)
        button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
        button.setFixedWidth(200)  # Set a fixed width for the buttons
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(button)

    # Add button layout to the main layout
    layout.addLayout(button_layout)
    
    def save_data():
        global signal_data
        if signal_data is None:
           output_box.append("No signal data loaded.")
           return
       
        options = QFileDialog.Options()
        output_file_path, _ = QFileDialog.getSaveFileName(None, "Save Signal Data", "", "Text Files (*.txt);;CSV Files (*.csv)", options=options)
        if not output_file_path:
           output_box.append("Save Data operation was cancelled.")
           return
        try:
            output_box.append("Saving Data....")
            fft_signal.save_signal_data(signal_data,output_file_path)
            output_box.append(f"Data Saved Successfully at {output_file_path}")
        except Exception as e:
            output_box.append(f"Error saving data: {e}")

    def plot_data():
        global signal_data
        if signal_data is None:
           output_box.append("No signal data loaded.")
           return
        frequencyspectrum   = getparameter.getfrequencyspectrum(signal_data)
        amplitudebuffer     = getparameter.getabsamplitude(signal_data)
        MaxAmplitude    = extract.getMaxAmplitude(amplitudebuffer)
        Totalpower      = extract.gettotalpower(amplitudebuffer)
        center_frequency,F1,F2 = getparameter.getBandPassFilterParameters(frequencyspectrum,MaxAmplitude,Totalpower)
        windowed_amplitudebuffer  = extract.applyHanningWindow(frequencyspectrum,amplitudebuffer,F1,F2)
        try:
             fig, axes = plt.subplots(1, 2, figsize=(12, 6))
             im1 = axes[0].imshow(amplitudebuffer, aspect='auto', cmap='viridis', origin='lower')
             axes[0].set_title("Original FFT Amplitude")
             axes[0].set_xlabel("Frequency Bin")
             axes[0].set_ylabel("Sample Index")
             fig.colorbar(im1, ax=axes[0], label="Amplitude")
             
             # Heatmap of windowed (filtered) FFT amplitude data
             im2 = axes[1].imshow(windowed_amplitudebuffer, aspect='auto', cmap='viridis', origin='lower')
             axes[1].set_title("Windowed FFT Amplitude")
             axes[1].set_xlabel("Frequency Bin")
             axes[1].set_ylabel("Sample Index")
             fig.colorbar(im2, ax=axes[1], label="Amplitude")
             
             plt.tight_layout()
             
             canvas = FigureCanvas(fig)
             toolbar = NavigationToolbar2QT(canvas)
             plot_dialog = QDialog()
             plot_dialog.setWindowTitle("Heat Map")
             dialog_layout = QVBoxLayout(plot_dialog)
             dialog_layout.addWidget(toolbar)
             dialog_layout.addWidget(canvas)
             plot_dialog.exec_()
             output_box.append("Data plotted successfully.")
        except Exception as e:
             output_box.append(f"Error while plotting data: {e}")
                
    def get_signal_characteristics():
        global signal_data
        if signal_data is None:
           output_box.append("No signal data loaded.")
           return
        
        frequencyspectrum   = getparameter.getfrequencyspectrum(signal_data)
        amplitudebuffer     = getparameter.getabsamplitude(signal_data)
        Fmax                = getparameter.getFmax(frequencyspectrum)
        Fmin                = getparameter.getFmin(frequencyspectrum)
        BW                  = getparameter.getBW(Fmax,Fmin)
        SamplingFrequency   = getparameter.getSamplingFrequency()
        FrequencyFactor     = getparameter.getfreqfactor(SamplingFrequency)
        FrequencyResolution = getparameter.getFreqresolution(BW)
        
        MaxAmplitude    = extract.getMaxAmplitude(amplitudebuffer)
        Totalpower      = extract.gettotalpower(amplitudebuffer)
       
        center_frequency,F1,F2 = getparameter.getBandPassFilterParameters(frequencyspectrum,MaxAmplitude,Totalpower)
        
        
        characteristics_dialog = QDialog()
        characteristics_dialog.setWindowTitle("Signal Characteristics")

        # Create a layout for the dialog
        dialog_layout = QVBoxLayout(characteristics_dialog)

        # Add labels to the layout to display the calculated values
        dialog_layout.addWidget(QLabel(f"Maximum Frequency (Fmax): {Fmax} Hz"))
        dialog_layout.addWidget(QLabel(f"Minimum Frequency (Fmin): {Fmin} Hz"))
        dialog_layout.addWidget(QLabel(f"Bandwidth (BW): {BW} Hz"))
        dialog_layout.addWidget(QLabel(f"Sampling Frequency: {SamplingFrequency} Hz"))
        dialog_layout.addWidget(QLabel(f"Frequency Factor: {FrequencyFactor}"))
        dialog_layout.addWidget(QLabel(f"Frequency Resolution: {FrequencyResolution} Hz"))
        dialog_layout.addWidget(QLabel(f"Centre Frequency: {center_frequency} Hz"))
        dialog_layout.addWidget(QLabel(f"Lower Cut-off Frequency: {F1} Hz"))
        dialog_layout.addWidget(QLabel(f"Upper Cut-off Frequency: {F2} Hz"))
        
        close_button = QPushButton("Close")
        close_button.setStyleSheet("font-size: 16px; padding: 5px;")
        close_button.clicked.connect(characteristics_dialog.close)
        dialog_layout.addWidget(close_button)  

        # Show the dialog
        characteristics_dialog.exec_()
        output_box.append("Signal characteristics displayed.")
    
    def get_signal_freqfeatures():
        global signal_data,file_path
        
        if signal_data is None:
           output_box.append("No signal data loaded.")
           return
        frequencyspectrum   = getparameter.getfrequencyspectrum(signal_data)
        amplitudebuffer     = getparameter.getabsamplitude(signal_data)
        
        MaxAmplitude    = extract.getMaxAmplitude(amplitudebuffer)
        Totalpower      = extract.gettotalpower(amplitudebuffer)
        center_frequency,F1,F2 = getparameter.getBandPassFilterParameters(frequencyspectrum,MaxAmplitude,Totalpower)
        amplitudebuffer = extract.applyHanningWindow(frequencyspectrum,amplitudebuffer,F1,F2)
        
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
        
        amplitudebuffer_list = [
                                MeanAmplitude, MaxAmplitude, PeaktoPeak, RMSAmplitude, Variance,
                                StdDev, Skewness, Kurtosis, Totalpower, crestfactor, formfactor,
                                peaktomeanratio, getmargin, RelativePekSpec]
        
            # List of feature names to use for titles
        feature_names = [
            "Mean Amplitude", "Max Amplitude", "Peak to Peak", "RMS Amplitude", "Variance",
            "Standard Deviation", "Skewness", "Kurtosis", "Total Power", "Crest Factor", "Form Factor",
            "Peak to Mean Ratio", "Margin", "Relative Peak Spectral"
        ]
        
        # Create the figure for plotting with increased size
        fig, axes = plt.subplots(4, 4, figsize=(18, 16))  # Increased figure size
        axes = axes.flatten() 
        
        # Set global font size for labels and titles
        plt.rcParams.update({'font.size': 14})  # Set font size for titles and labels

        for idx, ax in enumerate(axes):
            if idx < len(amplitudebuffer_list):
                feature = amplitudebuffer_list[idx]
                feature_name = feature_names[idx]  # Get corresponding feature name
                
                if isinstance(feature, pd.DataFrame) and len(feature.shape) == 2:
                    ax.plot(frequencyspectrum, feature.T)
                    ax.set_title(f"{feature_name} (DataFrame)", fontsize=16)
                else:
                    ax.plot(frequencyspectrum, feature)
                    ax.set_title(f"{feature_name}", fontsize=16)
                
                ax.set_xlabel("Frequency (Hz)", fontsize=14)  # Increase font size for x-axis
                ax.set_ylabel("Amplitude", fontsize=14)      # Increase font size for y-axis
            else:
                ax.axis('off')  # Turn off unused subplots if there are fewer than 16 plots

        # Apply tight_layout to avoid label overlapping
        plt.tight_layout(pad=4.0)  # Adjust the padding between plots

        # Create the canvas and toolbar for the dialog
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar2QT(canvas)

        # Create a dialog window to display the plot
        plot_dialog = QDialog()
        plot_dialog.setWindowTitle(f"Plot Data - {file_path if file_path else 'Plot Data'}")
        
        # Set window flags to allow maximize, minimize, and close buttons
        plot_dialog.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Set the dialog to be resizable and maximizable
        plot_dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_dialog.setMinimumSize(800, 600)
        
        # Layout for the dialog
        dialog_layout = QVBoxLayout(plot_dialog)
        dialog_layout.addWidget(toolbar) 
        dialog_layout.addWidget(canvas)  
        plot_dialog.exec_()
        output_box.append("All the Features from Frequency Domain successfully extracted..!! ")
        
    def get_signal_PCA():
        global signal_data,file_path
        if signal_data is None:
           output_box.append("No signal data loaded.")
           return
        
        frequencyspectrum   = getparameter.getfrequencyspectrum(signal_data)
        amplitudebuffer     = getparameter.getabsamplitude(signal_data)
        
        MaxAmplitude    = extract.getMaxAmplitude(amplitudebuffer)
        Totalpower      = extract.gettotalpower(amplitudebuffer)
        center_frequency,F1,F2 = getparameter.getBandPassFilterParameters(frequencyspectrum,MaxAmplitude,Totalpower)
        amplitudebuffer = extract.applyHanningWindow(frequencyspectrum,amplitudebuffer,F1,F2)
        
        PCAResult = extract.getfreqPCA(frequencyspectrum,amplitudebuffer)

        # Plotting the PCA result
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, 86), PCAResult.flatten(), marker='o', linestyle='-', color='r', alpha=0.7)
        ax.set_xlabel("Frequency Bin Index")
        ax.set_ylabel("PCA Component Value")
        ax.set_title("PCA Across Frequency Bins (Transposed 1x85)")
        ax.grid()
        
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar2QT(canvas)
        
        # Create a dialog window to show the plot
        plot_dialog = QDialog()
        plot_dialog.setWindowTitle(f"Plot Data - {file_path if file_path else 'Plot Data'}")
        
        # Set window flags to allow maximize, minimize, and close buttons
        plot_dialog.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # Set the dialog to be resizable and maximizable
        plot_dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_dialog.setMinimumSize(800, 600)
        
        # Layout for the dialog
        dialog_layout = QVBoxLayout(plot_dialog)
        dialog_layout.addWidget(toolbar)  # Add the toolbar for plot navigation
        dialog_layout.addWidget(canvas)   # Add the canvas to show the plot
        
        # Show the plot dialog
        plot_dialog.exec_()

        output_box.append("PCA Successfully Generated for the given Data..!!")

    # Connect buttons to their respective functions
    button_layout.itemAt(0).widget().clicked.connect(save_data)     
    button_layout.itemAt(1).widget().clicked.connect(plot_data)
    button_layout.itemAt(2).widget().clicked.connect(get_signal_characteristics)
    button_layout.itemAt(3).widget().clicked.connect(get_signal_freqfeatures)
    button_layout.itemAt(4).widget().clicked.connect(get_signal_PCA)
    
    # Add a spacer or title block above the new buttons section
    spacer = QSpacerItem(QSizePolicy.Minimum, QSizePolicy.Expanding)
    layout.addItem(spacer)  # Adds vertical spac
    
    # Optional: Add a block of text above the new buttons
    trainingmodelspacer_title = QLabel("Additional Controls")
    trainingmodelspacer_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    layout.addWidget(trainingmodelspacer_title)
    
    Train_button_layout = QHBoxLayout()
    Train_button_layout.setSpacing(10)  # Spacing between buttons
    Train_button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
    Train_button_layout.setAlignment(Qt.AlignLeft)
    
    # Add new buttons
    select_folder_button = QPushButton("  Select Folder  ")
    select_folder_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    select_folder_button.setFixedWidth(200)
    Train_button_layout.addWidget(select_folder_button)

    folder_path_label = QLabel("No folder selected")
    folder_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    folder_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    Train_button_layout.addWidget(folder_path_label)
    
     # Function to open folder dialog and set path
    def select_folder():
        global folder_path
        folder_path = QFileDialog.getExistingDirectory(
            None, "Select Folder"
        )
        if folder_path:
            folder_path_label.setText(folder_path)
            output_box.append(f"Folder selected: {folder_path}")

    # Connect the select folder button to the select folder function
    select_folder_button.clicked.connect(select_folder)
    
    layout.addLayout(Train_button_layout)
    
    # Add the new buttons
    Train_buttons = ["Train Model", "Model Performance"]
    for trainbutton_name in Train_buttons:
        new_button = QPushButton(trainbutton_name)
        new_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
        new_button.setFixedWidth(200) 
        new_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        Train_button_layout.addWidget(new_button)
    
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)  # Set min and max range
    progress_bar.setValue(0)  # Initial value
    progress_bar.setTextVisible(True)  # Display text inside the progress bar
    progress_bar.setStyleSheet(
        "QProgressBar {"
        "    border: 2px solid grey;"
        "    border-radius: 5px;"
        "    background-color: #f5f5f5;"
        "    text-align: center;"
        "}"
        "QProgressBar::chunk {"
        "    background-color: #4caf50;"
        "    border-radius: 5px;"
        "}"
    )
    
    # Layout to include progress bar and the percentage label
    layout.addWidget(progress_bar)
        
    def train_my_model():
        global folder_path, model , history
        
        if folder_path is None:
           output_box.append("No Folder or Files to Train...!!!")
           return
        
        ffttrain = FFTModel()

        ffttrain.load_fft_signals(folder_path)

        hard_signals = ffttrain.myfft_signals["Hard"]
        soft_signals = ffttrain.myfft_signals["Soft"]
        hard_pca_results, soft_pca_results = ffttrain.extractPCA(hard_signals, soft_signals)

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
        
        output_box.append(f"Combined Data (X) Shape: {X.shape}")
        output_box.append(f"Combined Labels (y) Shape: {y.shape}")

        # Check the combined data and labels
        print("Combined Data (X) Shape:", X.shape)
        print("Combined Labels (y) Shape:", y.shape)

        #model, history = ffttrain.CnnModelPCA(X, y)
        
        # Step 1: Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        output_box.append(f"X_train shape: {X_train.shape}")
        output_box.append(f"y_train shape: {y_train.shape}")

        # Step 2: Reshape the data for CNN input (samples, features, 1)
        X_train = X_train[..., np.newaxis]  # Add channel dimension
        X_test = X_test[..., np.newaxis]  # Same for the test data

        # Step 3: Dynamic batch size handling
        batch_size = min(32, X_train.shape[0])

        # Step 4: Build the CNN Model
        model = models.Sequential(
            [
                layers.Conv1D(
                    32,
                    3,
                    activation="relu",
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                ),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 3, activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.5),  # Add dropout to avoid overfitting
                layers.Dense(64, activation="relu"),
                layers.Dense(
                    1, activation="sigmoid"
                ),  # Binary classification (hard vs soft)
            ]
        )

        # Step 5: Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is not None:
                    progress_bar.setValue(int((epoch + 1) / 50 * 100))  # Assuming 50 epochs

        # Step 6: Print model architecture
        output_box.append("CNN Model Summary:")
        model.summary(print_fn=lambda x, line_break=True: output_box.append(x))
        
        # Step 7: Callbacks for training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        )

        # Step 8: Train the model
        output_box.append("Training the model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, lr_scheduler,ProgressBarCallback()],
        )

        # Step 9: Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        output_box.append(f"Model Test Loss: {test_loss:.4f}")
        output_box.append(f"Model Test Accuracy: {test_accuracy * 100:.2f}%")
        output_box.append("Model Successfully Trained...!!!") 
        
    def plot_model():
        global folder_path, model , history
        
        if history is None:
            output_box.append("Error: No training history available. Train the model first.")
            return
        
        # Create a new figure for the plot
        fig, ax = plt.subplots(figsize=(8, 5))  
        ax.plot(history.history["accuracy"], label="Train Accuracy")
        ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.set_title("Training History")
        
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar2QT(canvas)

        # Create a dialog window to show the plot
        plot_dialog = QDialog()
        plot_dialog.setWindowTitle("Model Training Plot")
        
        # Set window flags to allow maximize, minimize, and close buttons
        plot_dialog.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
         # Set the dialog to be resizable and maximizable
        plot_dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_dialog.setMinimumSize(800, 600)
        
        # Layout for the dialog
        dialog_layout = QVBoxLayout(plot_dialog)
        dialog_layout.addWidget(toolbar)  # Add the toolbar for plot navigation
        dialog_layout.addWidget(canvas)   # Add the canvas to show the plot
        
        # Show the plot dialog
        plot_dialog.exec_()
        output_box.append("Trained Model Graph")
        
    Train_button_layout.itemAt(2).widget().clicked.connect(train_my_model)
    Train_button_layout.itemAt(3).widget().clicked.connect(plot_model)
    
    save_model_button = QPushButton("Save Model")
    save_model_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    save_model_button.setFixedWidth(200)
    save_model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    Save_button_layout = QVBoxLayout()  # This layout will be placed below Train_button_layout
    Save_button_layout.addWidget(save_model_button)
    
    layout.addLayout(Save_button_layout)
    
    def save_trained_model():
        global model
        if model is None:
            output_box.append("Error: No trained model found. Train the model first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(None, "Save Model", "", "H5 Files (*.h5);;All Files (*)")
        
        if file_path:  # If user selects a file path
            model.save(file_path)
            output_box.append(f"Model saved successfully at: {file_path}")
    
    save_model_button.clicked.connect(save_trained_model)
    
    # Add a spacer or title block above the new buttons section
    spacer = QSpacerItem(QSizePolicy.Minimum, QSizePolicy.Expanding)
    layout.addItem(spacer)  # Adds vertical spac
    
    # Optional: Add a block of text above the new buttons
    predictmodelspacer_title = QLabel("Predict Model")
    predictmodelspacer_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    layout.addWidget(predictmodelspacer_title)
    
    Predict_button_layout = QHBoxLayout()
    Predict_button_layout.setSpacing(10)  # Spacing between buttons
    Predict_button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
    Predict_button_layout.setAlignment(Qt.AlignLeft)
    
    # Add new buttons
    select_predictmodel_button = QPushButton("  Select Model  ")
    select_predictmodel_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    select_predictmodel_button.setFixedWidth(200)
    Predict_button_layout.addWidget(select_predictmodel_button)

    model_path_label = QLabel("No File selected")
    model_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    model_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    Predict_button_layout.addWidget(model_path_label)
    
    # -------- Select File Button --------
    select_predictfile_button = QPushButton("  Select File  ")
    select_predictfile_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    select_predictfile_button.setFixedWidth(200)
    Predict_button_layout.addWidget(select_predictfile_button)
    
    # File Path Label
    predictfile_path_label = QLabel("No File Selected")
    predictfile_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    predictfile_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    Predict_button_layout.addWidget(predictfile_path_label)
    
    predictselectedfile_button = QPushButton("Predict")
    predictselectedfile_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    predictselectedfile_button.setFixedWidth(200)
    predictselectedfile_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    Predict_button_layout.addWidget(predictselectedfile_button)
    
    # Function to open folder dialog and set path
    def select_predictionmodel():
        global predictmodelpath,Loadcnnmodel
        Predict = Predict_FFT()
        predictmodelpath, _ = QFileDialog.getOpenFileName(None, "Select Model", "", "H5 Files (*.h5);;All Files (*)")
        if predictmodelpath:
           Loadcnnmodel = Predict.loadCNNModel(predictmodelpath)
           model_path_label.setText(os.path.basename(predictmodelpath))
           output_box.append(f"Model selected: {predictmodelpath}")
        else:
           output_box.append("Invalid Model or File..!!")
           
    def select_predictfile():
        global predictmodelpath,predictfilepath
        predictfilepath, _ = QFileDialog.getOpenFileName(None, "Select File", "", "Txt Files (*.txt);;All Files (*)")
        if predictfilepath:
            predictfile_path_label.setText(os.path.basename(predictfilepath))
            output_box.append(f"File selected: {predictfilepath}")
        else:
            output_box.append("Invalid File..!!")
    
    def show_prediction_dialog(message,Path):
        global predictmodelpath,predictfilepath
        dialog = QDialog()
        dialog.setWindowTitle("Prediction Result")
        dialog.setMinimumSize(400, 300)
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        LOGO_DIR = os.path.join(BASE_DIR, "Logo")
        IMAGE_PATH = os.path.join(LOGO_DIR, Path)
        print(IMAGE_PATH)

        # Load and display image
        image_label = QLabel()
        pixmap = QPixmap(IMAGE_PATH)
        pixmap = pixmap.scaled(250, 250)  # Resize if needed
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)

        # Display message
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; text-align: center;")

        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)
        layout.addWidget(message_label)
        dialog.setLayout(layout)
        predicted_labels = []  # Clear stored predictions
        dialog.exec_()
            
    def predict_selectedfile():
        global predictmodelpath,predictfilepath,Loadcnnmodel
        
        Predict = Predict_FFT()
        PredictPCA = Predict.loadpredictfile(predictfilepath)

        UpdatedPredictPCA = PredictPCA.reshape(PredictPCA.shape[0], -1)

        predictions = Loadcnnmodel.predict(UpdatedPredictPCA)

        predicted_labels = (predictions > 0.5).astype(int)
        
        for label in predicted_labels:
            if label == 0:
                result_message = "Wear Seat Belt...!!"
                output_box.append("Wear Seat Belt...!!")
                Image_Path = "Warning.png"
            else:
                result_message = "No Presence Detected.."
                output_box.append(f"No Presence Detected..")
                Image_Path = "No Warning.png"
        show_prediction_dialog(result_message,Image_Path)
        
    # Connect the select folder button to the select folder function
    select_predictmodel_button.clicked.connect(select_predictionmodel)
    select_predictfile_button.clicked.connect(select_predictfile)
    predictselectedfile_button.clicked.connect(predict_selectedfile)
    
    layout.addLayout(Predict_button_layout)