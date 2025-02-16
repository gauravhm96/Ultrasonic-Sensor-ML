from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QRadioButton, QSizePolicy,
                             QStackedWidget, QVBoxLayout, QWidget,QDialog,QProgressBar,QApplication,QTextEdit,QGridLayout,QSpacerItem)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt

from SignalProcess import SignalProcessor
from ObjectDetectionFeatureExtract import FeatureExtract
from ObjectDetectionTraining import BinaryImageClassifier

from adc_signal_process import ADCSignal,ADCSignalProcess,TrainADC,PredictADC

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# ------------------- Version 1 Global Variables -------------------
V1_FILE_PATH = None
v1_adc_data = None
shared_data = {}



# ------------------- Version 2 Global Variables -------------------
V2_FILE_PATH = None
FOLDER_PATH = None
PREDICT_MODEL_PATH = None
PREDICT_FILE_PATH = None

my_adc_data = None
filtered_myadc_data = None
peak_index = None
peak_value = None
dist = None
model = None
Loadmodel = None
X_test = None 
y_test = None
# ------------------- Version 2 Global Variables -------------------

def object_detection_features(layout, output_box):
    
    
    # ------------------- Version 1 Function Declaration -------------------
    signalprocess = SignalProcessor()
    extractfeature = FeatureExtract()
    
    
    # ------------------- Version 2 Function Declaration -------------------
    myadcdata = ADCSignal()
    processadcdata = ADCSignalProcess()
    train = TrainADC()
    
    
    # Create a horizontal layout for the version selection section
    detection_label = QLabel("Select version for Object Detection Algoritm")
    detection_label.setStyleSheet("font-size: 18px; font-weight: normal;padding: 5px")
    layout.addWidget(detection_label)

    version_layout = QHBoxLayout()
    version_layout.setSpacing(10)
    version_layout.setContentsMargins(0, 0, 0, 0)
    version_layout.setAlignment(Qt.AlignLeft)

    # Spacer title label "Select Version"
    select_label = QLabel("Select Version:")
    select_label.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    select_label.setFixedWidth(200)
    select_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    version_layout.addWidget(select_label)

    # Create radio buttons for version selection
    radio_v1 = QRadioButton("v0.1")
    radio_v2 = QRadioButton("v0.2")
    radio_v1.setChecked(True)  # Default selection

    # Add the radio buttons to the layout
    version_layout.addWidget(radio_v1, alignment=Qt.AlignLeft)
    version_layout.addWidget(radio_v2, alignment=Qt.AlignLeft)

    # Add the version selection layout to the main layout
    layout.addLayout(version_layout)

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setLineWidth(2)
    layout.addWidget(line)

    # Create a QStackedWidget to hold different layouts for each version
    stacked_widget = QStackedWidget()

    # ------------------- Version 1 Layout -------------------
    version1_widget = QWidget()
    v1_layout = QVBoxLayout()
    v1_layout.setSpacing(10)
    v1_layout.setContentsMargins(0, 0, 0, 0)
    v1_layout.setAlignment(Qt.AlignTop)

    label_v1 = QLabel("Object Detection Algoritm Type 1")
    label_v1.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    v1_layout.addWidget(label_v1)
    
    def v1_select_file():
        global v1_adc_data, V1_FILE_PATH
        V1_FILE_PATH, _ = QFileDialog.getOpenFileName(
            None, "Select ADC File", "", "ADC Data (*.txt);;CSV (*.csv)"
        )
        if V1_FILE_PATH:
            if "adc" not in V1_FILE_PATH.lower():
                output_box.append("I think you have not selected ADC Data :( ")
                return
            else:
                v1_file_path_label.setText(V1_FILE_PATH)
                output_box.append(f"File selected: {V1_FILE_PATH}")
                try:
                    signalprocess.set_file_path(V1_FILE_PATH)
                    v1_adc_data = signalprocess.load_signal_data()
                    output_box.append("ADC Data Loaded Successfully.")
                    output_box.append(f"ADC Data:\n{v1_adc_data[:10]}")
                except Exception as e:
                    output_box.append(f"Error loading ADC data: {e}")
    
    v1_file_layout = QHBoxLayout()
    v1_file_layout.setSpacing(10)
    v1_file_layout.setContentsMargins(0, 0, 0, 0)
    v1_file_layout.setAlignment(Qt.AlignLeft)
    
    v1_select_file_button = QPushButton("Select ADC Data")
    v1_select_file_button.setStyleSheet(
        "font-size: 18px;font-weight: normal; padding: 5px;"
    )
    v1_select_file_button.setFixedWidth(200)
    v1_select_file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    v1_select_file_button.clicked.connect(v1_select_file)
    v1_file_layout.addWidget(v1_select_file_button)

    # Add a QLabel to show the file path
    v1_file_path_label = QLabel("No file selected")
    v1_file_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    v1_file_layout.addWidget(v1_file_path_label)
    v1_layout.addLayout(v1_file_layout)
    
    # Create a horizontal layout for all buttons
    v1_calculatedist_layout = QHBoxLayout()
    v1_calculatedist_layout.setSpacing(10)  
    v1_calculatedist_layout.setContentsMargins(0, 0, 0, 0) 
    v1_calculatedist_layout.setAlignment(Qt.AlignLeft)
    
    v1_calculatedist_button = QPushButton("Find First Echo")
    v1_calculatedist_button.setStyleSheet(
        "font-size: 18px;font-weight: normal; padding: 5px;"
    )
    v1_calculatedist_button.setFixedWidth(200)
    v1_calculatedist_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    def v1_calculate_distance():
        global v1_adc_data, V1_FILE_PATH
        if v1_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        try:
            v1_progress_bar.setValue(0)
            QApplication.processEvents()
            if signalprocess:
               # Process the signal
               signalprocess.analyze_raw_signals()
               v1_progress_bar.setValue(25)
               signalprocess.annotate_real_peaks()
               v1_progress_bar.setValue(50)
               signalprocess.NoiseFiltering()
               v1_progress_bar.setValue(75)
               signalprocess.SignalCorrection()
               v1_progress_bar.setValue(90)
               
               # Calculate the distance
               distance_info = signalprocess.Calculate_Distance()
               
                # Extract the distance value from the returned dictionary
               distance = distance_info.get("distance", 0.0)
               
               v1_progress_bar.setValue(100)
               
               # Update the label with the calculated distance
               v1_dist_label.setText(f"{distance:.2f} m")
               v1_dist_label.setStyleSheet("font-size: 18px; padding: 5px; border: 3px solid black;background-color: green;")
               output_box.append("Distance calculation completed successfully.")
               
        except Exception as e:
            output_box.append(f"Error during distance calculation: {e}")
            
    v1_calculatedist_button.clicked.connect(v1_calculate_distance)
    
    v1_calculatedist_layout.addWidget(v1_calculatedist_button)
    
    v1_progress_bar = QProgressBar()
    v1_progress_bar.setRange(0, 100)  # Set min and max range
    v1_progress_bar.setValue(0) 
    v1_progress_bar.setTextVisible(True)  # Display text inside the progress bar
    v1_progress_bar.setStyleSheet(
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
    v1_calculatedist_layout.addWidget(v1_progress_bar)
    
    v1_showcalculation_button = QPushButton("Show Computation")
    v1_showcalculation_button.setStyleSheet(
        "font-size: 18px;font-weight: normal; padding: 5px;"
    )
    v1_showcalculation_button.setFixedWidth(200)
    v1_showcalculation_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
    def v1_show_calculation():
        global v1_adc_data, V1_FILE_PATH
        v1_progress_bar.setValue(0) 
        
        if v1_adc_data is None:
            output_box.append("No ADC data loaded.")
            return
        try:
            dialog = QDialog()
            dialog.setWindowTitle("Computation Results")
            dialog.setFixedSize(2400, 1250)
            
            # Main layout with margins and spacing
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(20, 20, 20, 20)
            main_layout.setSpacing(20)
            
            # Title Label
            title_label = QLabel("Calculations")
            title_label.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px;")
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)
            
            # Description Label
            description_label = QLabel("Distance calculation and signal processing details.")
            description_label.setStyleSheet("font-size: 18px; padding: 10px;")
            description_label.setWordWrap(True)
            main_layout.addWidget(description_label)
            
            # Grid layout for plots
            plots_layout = QGridLayout()
            plots_layout.setSpacing(20)
            plots_layout.setContentsMargins(20, 20, 20, 20)
            
            # Get the figures and canvases from your signal processing functions
            fig1, axs1 = signalprocess.PlotRawSignal()
            fig1.set_size_inches(10, 6)
            canvas1 = FigureCanvas(fig1)
            
            fig2, axs2 = signalprocess.PlotNoiseFilteredSignal()
            fig2.set_size_inches(10, 6)
            canvas2 = FigureCanvas(fig2)
            
            fig3, axs3 = signalprocess.PlotSignalCorrection()
            fig3.set_size_inches(20, 6)
            canvas3 = FigureCanvas(fig3)
            
            # Place canvases in the grid (2 rows x 2 columns; third plot spans both columns)
            plots_layout.addWidget(canvas1, 0, 0)
            plots_layout.addWidget(canvas2, 0, 1)
            plots_layout.addWidget(canvas3, 1, 0, 1, 2)
            
            main_layout.addLayout(plots_layout)
            
            # Bottom section with a spacer and a centered Close button
            bottom_layout = QHBoxLayout()
            bottom_layout.addStretch()
            close_button = QPushButton("Close")
            close_button.setFixedSize(150, 40)
            close_button.setStyleSheet("font-size: 18px; padding: 5px;")
            close_button.clicked.connect(dialog.close)
            bottom_layout.addWidget(close_button)
            bottom_layout.addStretch()
            main_layout.addLayout(bottom_layout)
            
            dialog.setLayout(main_layout)
            dialog.exec()
        except Exception as e:
            output_box.append(f"Error during Rendering calculation Output: {e}")

    
    v1_showcalculation_button.clicked.connect(v1_show_calculation)
    
    v1_calculatedist_layout.addWidget(v1_showcalculation_button)
    
    v1_dist_label = QLabel("0.0 cm")
    v1_dist_label.setStyleSheet("font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;")

    v1_calculatedist_layout.addWidget(v1_dist_label)
    
    v1_layout.addLayout(v1_calculatedist_layout)
    
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setLineWidth(2)
    v1_layout.addWidget(line)
    
    v1_trainingmodelspacer_title = QLabel("Train Model")
    v1_trainingmodelspacer_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    v1_layout.addWidget(v1_trainingmodelspacer_title)
    
    v1_train_layout = QHBoxLayout()
    v1_train_layout.setSpacing(10)
    v1_train_layout.setContentsMargins(0, 0, 0, 0)
    v1_train_layout.setAlignment(Qt.AlignLeft)
    
    v1_select_trainfile_button = QPushButton("Select ADC Data")
    v1_select_trainfile_button.setStyleSheet(
        "font-size: 18px;font-weight: normal; padding: 5px;"
    )
    v1_select_trainfile_button.setFixedWidth(200)
    v1_select_trainfile_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    def v1_select_file_train():
        global v1_adc_data, V1_FILE_PATH
        V1_FILE_PATH, _ = QFileDialog.getOpenFileName(
            None, "Select ADC File", "", "ADC Data (*.txt);;CSV (*.csv)"
        )
        if V1_FILE_PATH:
            if "adc" not in V1_FILE_PATH.lower():
                output_box.append("I think you have not selected ADC Data :( ")
                return
            else:
                v1_trainfile_path_label.setText(os.path.basename(V1_FILE_PATH))
                output_box.append(f"File selected: {V1_FILE_PATH}")
                try:
                    signalprocess.set_file_path(V1_FILE_PATH)
                    v1_adc_data = signalprocess.load_signal_data()
                    output_box.append("ADC Data Loaded Successfully.")
                    output_box.append(f"ADC Data:\n{v1_adc_data[:10]}")
                except Exception as e:
                    output_box.append(f"Error loading ADC data: {e}")
        
    v1_select_trainfile_button.clicked.connect(v1_select_file_train)
    v1_train_layout.addWidget(v1_select_trainfile_button)

    # Add a QLabel to show the file path
    v1_trainfile_path_label = QLabel("No file selected")
    v1_trainfile_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    v1_train_layout.addWidget(v1_trainfile_path_label)
    
    # Add the required buttons
    v1_trainbuttons = ["Analyze Signal", "Type 1-Spectgrm Peak", "Type 2-Spectgrm Peak Spect", "Spectgrm Non-Peak"]
    for button_name in v1_trainbuttons:
        button = QPushButton(button_name)
        button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
        button.setFixedWidth(250)  # Set a fixed width for the buttons
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        v1_train_layout.addWidget(button)
        
    def analyze_signal():
        global v1_adc_data,shared_data
        if v1_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        else:
            try:
                output_box.append("Signal analysis started...")
                
                signalprocess.load_signal_data()
                v1_trainprogress_bar.setValue(20)  # Update progress to 20%
                output_box.append("Signal data loaded.")
                
                signalprocess.analyze_raw_signals()
                v1_trainprogress_bar.setValue(20)  # Update progress to 40%
                output_box.append("Raw signals analyzed.")
                
                signalprocess.annotate_real_peaks()
                v1_trainprogress_bar.setValue(40)  # Update progress to 50%
                output_box.append("Real peaks annotated.")
                
                updated_signals, threshold_info = signalprocess.NoiseFiltering()
                v1_trainprogress_bar.setValue(50)  # Update progress to 60%
                output_box.append("Noise filtering applied.")        
                output_box.append(f"Updated Signals: {updated_signals}")
                output_box.append(f"Threshold Info: {threshold_info}")
                
                updated_signals, overall_threshold = extractfeature.apply_threshold_filtering(updated_signals)
                v1_trainprogress_bar.setValue(60)  # Update progress to 70%
                output_box.append("Threshold filtering applied.")
                output_box.append(f"Updated Signals: {updated_signals}")
                output_box.append(f"Threshold Info: {overall_threshold}")
                shared_data['updated_signals'] = updated_signals
                
                selected_peak_windows = extractfeature.extract_peak_windows(updated_signals)
                v1_trainprogress_bar.setValue(70)  # Update progress to 80%
                output_box.append("Peak windows extracted.")
                output_box.append(f"Updated Signals: {selected_peak_windows}")
                shared_data['selected_peak_windows'] = selected_peak_windows
                
                selected_non_peak_windows = extractfeature.extract_non_peak_windows(updated_signals)
                v1_trainprogress_bar.setValue(80) # Update progress to 90%
                output_box.append(f"Updated Signals: {selected_non_peak_windows}")
                shared_data['selected_non_peak_windows'] = selected_non_peak_windows        
                
                window_duration,ADC_SAMPLE_FREQUENCY = extractfeature.calulate_window(num_samples_per_window=300)
                v1_trainprogress_bar.setValue(95)  # Update progress to 90%
                output_box.append("window_duration Determined!")      
                output_box.append(f"Window duration: {window_duration}, ADC Sample Frequency: {ADC_SAMPLE_FREQUENCY}")
                shared_data['ADC_SAMPLE_FREQUENCY'] = ADC_SAMPLE_FREQUENCY
                
                output_box.append("Feature Extraction Completed..!!")
                v1_trainprogress_bar.setValue(100)  # Update progress to 100%
                
            except Exception as e:
                output_box.append(f"An error occurred while Analyzing data: {e}")
                
    
    def Generate_Spectogram_PeaksType1():
        v1_trainprogress_bar.setValue(0)
        global v1_adc_data,shared_data
        if v1_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        else:
            try:
                output_box.append("Generating Spectrograms...")
               
                folder_path = os.path.dirname(os.path.abspath(__file__))
                PeakSpectrogramType1 = os.path.join(folder_path, "PeakspectrogramType1")
        
                if not os.path.exists(PeakSpectrogramType1):
                   os.makedirs(PeakSpectrogramType1)
                   output_box.append(f"Folder 'Peakspectrogram' created at {PeakSpectrogramType1}")
                else:
                   output_box.append(f"Folder 'Peakspectrogram' already exists at {PeakSpectrogramType1}")
        
                   output_box.append("Preparing to save spectrograms...")
                   v1_trainprogress_bar.setValue(10)
        
                   output_box.append("Processing selected peaks...")
        
                   v1_trainprogress_bar.setValue(30)
                   QApplication.processEvents()
        
                   extractfeature.save_PeakSspectrogramsType_1(shared_data['selected_peak_windows'],
                                                    shared_data['updated_signals'],
                                                    PeakSpectrogramType1,
                                                    num_samples_per_window= 300,ADC_SAMPLE_FREQUENCY=shared_data["ADC_SAMPLE_FREQUENCY"]
                                                    )
        
                   output_box.append("Saving spectrograms...")
                   v1_trainprogress_bar.setValue(100)
                   output_box.append("Spectrograms Generated...!!")
            except Exception as e:
                output_box.append(f"An error occurred while Generating Spectograms: {e}")

    def Generate_Spectogram_PeaksType2():
        v1_trainprogress_bar.setValue(0)
        global v1_adc_data,shared_data
        if v1_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        else:
            try:
                output_box.append("Generating Spectrograms...")
                    
                folder_path = os.path.dirname(os.path.abspath(__file__))
                PeakSpectrogramType2 = os.path.join(folder_path, "PeakspectrogramType2")
                
                if not os.path.exists(PeakSpectrogramType2):
                   os.makedirs(PeakSpectrogramType2)
                   output_box.append(f"Folder 'Peakspectrogram' created at {PeakSpectrogramType2}")
                else:
                   output_box.append(f"Folder 'Peakspectrogram' already exists at {PeakSpectrogramType2}")
                   
                output_box.append("Preparing to save spectrograms...")
                v1_trainprogress_bar.setValue(10)
                
                output_box.append("Processing selected peaks...")
                
                v1_trainprogress_bar.setValue(30)
                QApplication.processEvents()
                    
                extractfeature.save_PeakSspectrogramsType_2(shared_data['selected_peak_windows'], PeakSpectrogramType2, figure_size = (3, 3))
                
                output_box.append("Saving spectrograms...")
                v1_trainprogress_bar.setValue(100)
                
                output_box.append("Spectrograms Generated...!!")
                    
            except Exception as e:
                output_box.append(f"An error occurred while Generating Spectograms: {e}")
                
    def Generate_Spectogram_NonPeaks():
        v1_trainprogress_bar.setValue(0)
        global v1_adc_data,shared_data
        if v1_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        else:
            try:
                output_box.append("Generating Spectrograms...")
                
                folder_path = os.path.dirname(os.path.abspath(__file__))
                NonPeakSpectrogram = os.path.join(folder_path, "NonPeakspectrogram")
                
                if not os.path.exists(NonPeakSpectrogram):
                    os.makedirs(NonPeakSpectrogram)
                    output_box.append(f"Folder 'Peakspectrogram' created at {NonPeakSpectrogram}")
                else:
                    output_box.append(f"Folder 'Peakspectrogram' already exists at {NonPeakSpectrogram}")
                    
                output_box.append("Preparing to save spectrograms...")
                v1_trainprogress_bar.setValue(10)
                
                output_box.append("Processing selected peaks...")
                
                v1_trainprogress_bar.setValue(30)

                QApplication.processEvents()
                    
                extractfeature.save_NonPeakSspectrograms(shared_data['selected_non_peak_windows'], NonPeakSpectrogram)
                
                output_box.append("Saving spectrograms...")
                v1_trainprogress_bar.setValue(100)
                
                output_box.append("Spectrograms Generated...!!")
                
            except Exception as e:
                output_box.append(f"An error occurred while Generating Spectograms: {e}")
    
    v1_train_layout.itemAt(2).widget().clicked.connect(analyze_signal)
    v1_train_layout.itemAt(3).widget().clicked.connect(Generate_Spectogram_PeaksType1)  
    v1_train_layout.itemAt(4).widget().clicked.connect(Generate_Spectogram_PeaksType2)  
    v1_train_layout.itemAt(5).widget().clicked.connect(Generate_Spectogram_NonPeaks)  

    v1_layout.addLayout(v1_train_layout)
    
    TrainingProgress_layout = QVBoxLayout()
    TrainingProgress_layout.setSpacing(10)
    TrainingProgress_layout.setContentsMargins(0, 0, 0, 0)
    TrainingProgress_layout.setAlignment(Qt.AlignTop)
    
    v1_trainprogress_bar = QProgressBar()
    v1_trainprogress_bar.setRange(0, 100)  # Set min and max range
    v1_trainprogress_bar.setValue(0) 
    v1_trainprogress_bar.setTextVisible(True)  # Display text inside the progress bar
    v1_trainprogress_bar.setStyleSheet(
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
    TrainingProgress_layout.addWidget(v1_trainprogress_bar)
    v1_layout.addLayout(TrainingProgress_layout)
    
    v1_trainsave_layout = QHBoxLayout()
    v1_trainsave_layout.setSpacing(10)
    v1_trainsave_layout.setContentsMargins(0, 0, 0, 0)
    v1_trainsave_layout.setAlignment(Qt.AlignLeft)
    
    # Add the required buttons
    v1_trainsavebuttons = ["Train Model", "View Model Performance", "Save Model"]
    for button_name in v1_trainsavebuttons:
        button = QPushButton(button_name)
        button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
        button.setFixedWidth(250)  # Set a fixed width for the buttons
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        v1_trainsave_layout.addWidget(button)
    
    v1_layout.addLayout(v1_trainsave_layout)
    
    def v1_train_model():
        v1_trainprogress_bar.setValue(0)
        try:
            output_box.append("Training model...")
            folder_path = os.path.dirname(os.path.abspath(__file__))
            PeakSpectrogram = os.path.join(folder_path, "PeakspectrogramType1")
            NonPeakSpectrogram = os.path.join(folder_path, "NonPeakspectrogram")
            
            output_box.append("Initializing BinaryImageClassifier...")
            classifier = BinaryImageClassifier(PeakSpectrogram, NonPeakSpectrogram)
            v1_trainprogress_bar.setValue(30)
            QApplication.processEvents()
            
            output_box.append("Loading and preparing training data...")
            x_train, x_val, y_train, y_val = classifier.load_and_prepare_data()
            v1_trainprogress_bar.setValue(45)
            QApplication.processEvents()
            
            output_box.append("Building model architecture...")
            classifier.build_model()
            v1_trainprogress_bar.setValue(55)
            QApplication.processEvents()
            
            output_box.append("Training model on dataset...")
            
            history = classifier.train_model(x_train, y_train, x_val, y_val)
            v1_trainprogress_bar.setValue(90)
            # Check if history is not None before printing training details
            if history is not None:
                # Assuming history is a dictionary with lists for 'loss' and 'accuracy'
                output_box.append(f"Final Training Loss: {history['loss'][-1]:.4f}")
                output_box.append(f"Final Training Accuracy: {history['accuracy'][-1]:.4f}")
            else:
                output_box.append("No training log available.")
            QApplication.processEvents()
            
            output_box.append("Evaluating model performance...")
            results = classifier.evaluate_model(x_val, y_val)
            output_box.append("Final Evaluation Results: " + str(results))
            v1_trainprogress_bar.setValue(100)  # Update progress to 100%

            output_box.append("Model training completed!")
            
        except Exception as e:
            output_box.append(f"An error occurred while Training Data: {e}")
                
    
    v1_trainsave_layout.itemAt(0).widget().clicked.connect(v1_train_model)
    
    
    

    version1_widget.setLayout(v1_layout)

    # ------------------- Version 2 Layout -------------------
    version2_widget = QWidget()
    v2_layout = QVBoxLayout()
    v2_layout.setSpacing(10)
    v2_layout.setContentsMargins(0, 0, 0, 0) 
    v2_layout.setAlignment(Qt.AlignTop) 

    label_v2 = QLabel("Object Detection Algoritm Type 2")
    label_v2.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    v2_layout.addWidget(label_v2)

    def select_file():
        global my_adc_data,V2_FILE_PATH
        V2_FILE_PATH, _ = QFileDialog.getOpenFileName(
            None, "Select ADC File", "", "ADC Data (*.txt);;CSV (*.csv)"
        )
        if V2_FILE_PATH:
            if "adc" not in V2_FILE_PATH.lower():
                output_box.append("I think you have not selected ADC Data :( ")
                return
            else:
                file_path_label.setText(V2_FILE_PATH)
                output_box.append(f"File selected: {V2_FILE_PATH}")
                try:
                    my_adc_data = myadcdata.get_adc_data(V2_FILE_PATH)
                    output_box.append("ADC Data Loaded Successfully.")
                    output_box.append(f"ADC Data:\n{my_adc_data[:10]}")
                except Exception as e:
                    output_box.append(f"Error loading ADC data: {e}")

    file_layout = QHBoxLayout()
    file_layout.setSpacing(10)
    file_layout.setContentsMargins(0, 0, 0, 0)
    file_layout.setAlignment(Qt.AlignLeft)

    # Button to trigger the dummy function for Version 2
    select_file_button = QPushButton("Select ADC Data")
    select_file_button.setStyleSheet(
        "font-size: 18px;font-weight: normal; padding: 5px;"
    )
    select_file_button.setFixedWidth(200)
    select_file_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    select_file_button.clicked.connect(select_file)  # Connect the dummy function
    file_layout.addWidget(select_file_button)

    # Add a QLabel to show the file path
    file_path_label = QLabel("No file selected")
    file_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )

    file_layout.addWidget(file_path_label)
    v2_layout.addLayout(file_layout)
    
    # Create a horizontal layout for all buttons
    button_layout = QHBoxLayout()
    button_layout.setSpacing(10)  
    button_layout.setContentsMargins(0, 0, 0, 0) 
    button_layout.setAlignment(Qt.AlignLeft)
    
    # Add the required buttons
    buttons = ["View ADC Data", "Perform FFT", "View Noiseless Data","Calculate Dist","View Peaks"]
    for button_name in buttons:
        button = QPushButton(button_name)
        button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
        button.setFixedWidth(200)   
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(button)

    v2_layout.addLayout(button_layout)
    
    def view_adc_data():
        global my_adc_data
        if my_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        try:
            plot_adc_data(my_adc_data)
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    def Perform_fft():
        global my_adc_data,V2_FILE_PATH,filtered_myadc_data
        if my_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        try:
            lowcut = 39500.0  # Lower cutoff frequency in Hz
            highcut = 41500.0  # Upper cutoff frequency in Hz
            fs = 1.953125e6  # Sampling frequency in Hz (as per RED Pitaya)
            sos = processadcdata.butter_bandpass(lowcut, highcut, fs)
            
            adc_data_array = my_adc_data.to_numpy()
            filtered_data_array = np.apply_along_axis(processadcdata.apply_bandpass_filter, 1, adc_data_array, sos)
            filtered_myadc_data = pd.DataFrame(filtered_data_array)
            
            freqs,avg_fft_magnitude,dynamic_threshold,f_start,f_end = processadcdata.performfft(filtered_myadc_data, fs)
            
            if f_start is not None:
               output_box.append("Start frequency: {:.2f} Hz".format(f_start))
            else:
               output_box.append("No frequency components exceed the threshold.") 
            
            if f_end is not None:
               output_box.append("End frequency: {:.2f} Hz".format(f_end))
            else:
               output_box.append("No frequency components exceed the threshold.") 
            
            fig = plt.figure(figsize=(10, 6))
            plt.plot(freqs, avg_fft_magnitude, label="Avg FFT Magnitude", color="blue")
            plt.axhline(y=dynamic_threshold,color="green",linestyle="--",label=f"Threshold ({dynamic_threshold*100:.0f}% of max)",)
            plt.axvline(x=f_start,color="red",linestyle="--",label=f"Start Frequency: {f_start:.0f} Hz",)
            plt.axvline(x=f_end,color="magenta",linestyle="--",label=f"End Frequency: {f_end:.0f} Hz",)
            plt.xlim(30000, 50000)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title("Averaged FFT Magnitude with Detected Signal Band")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar2QT(canvas)
            plot_dialog = QDialog()
            plot_dialog.setWindowTitle(f"{V2_FILE_PATH}")
            dialog_layout = QVBoxLayout(plot_dialog)
            dialog_layout.addWidget(toolbar)
            dialog_layout.addWidget(canvas)
            plot_dialog.exec_()
            
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    def view_noiseless_data():
        global my_adc_data,filtered_myadc_data
        if my_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        if filtered_myadc_data is None:
           output_box.append("ADC Data is not Noise Filtered")
           return
        try:
            plot_adc_data(filtered_myadc_data)
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    def calculate_dist():
        global my_adc_data,filtered_myadc_data,peak_index, peak_value,dist
        if my_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        if filtered_myadc_data is None:
           output_box.append("ADC Data is not Noise Filtered")
           return
        try:
            peak_index, peak_value = processadcdata.detect_prominent_peaks(filtered_myadc_data,std_multiplier=3)
            output_box.append(f"Highest Peak Index: {peak_index}")
            output_box.append(f"Highest Peak Amplitude: {peak_value}")
            
            dist = (processadcdata.calculate_distance_from_peak(peak_index))*100
            dist_label.setText(f"{dist:.2f} cm")
            output_box.append(f"First Echo Detected at Dist: {dist}cm")
            
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    def plot_adc_data(adc_data):
        global V2_FILE_PATH
        if adc_data is None or adc_data.empty:
            print("No Data Available")
            return
        try:
            fig = plt.figure(figsize=(10, 6))
            for i in range(adc_data.shape[0]):
                plt.plot(adc_data.columns, adc_data.iloc[i, :], label=f"Row {i + 1}")
            plt.title("ADC Signal Data")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar2QT(canvas)
            plot_dialog = QDialog()
            plot_dialog.setWindowTitle(f"{V2_FILE_PATH}")
            dialog_layout = QVBoxLayout(plot_dialog)
            dialog_layout.addWidget(toolbar)
            dialog_layout.addWidget(canvas)
            plot_dialog.exec_()
            output_box.append("Data Loaded Successfully..!!") 
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    def view_peaks():
        global my_adc_data,V2_FILE_PATH,filtered_myadc_data,peak_index,peak_value
        if my_adc_data is None:
           output_box.append("No ADC data loaded.")
           return
        if filtered_myadc_data is None:
           output_box.append("ADC Data is not Noise Filtered")
           return
        try:
           fig = plt.figure(figsize=(10, 6))
           for index, row in filtered_myadc_data.iterrows():plt.plot(row, color="lightgray", linewidth=0.5, alpha=0.7)
           plt.axvline(x=peak_index, color="red", linestyle="--", label=f"Peak Index: {peak_index}")
           plt.axhline(y=peak_value, color="blue", linestyle="--", label=f"Peak Amplitude: {peak_value}")
           plt.title("All Filtered ADC Signals with Highlighted Peak")
           plt.xlabel("Sample Index")
           plt.ylabel("Amplitude")
           plt.legend()
           plt.grid(True)
           plt.tight_layout()
           canvas = FigureCanvas(fig)
           toolbar = NavigationToolbar2QT(canvas)
           plot_dialog = QDialog()
           plot_dialog.setWindowTitle(f"{V2_FILE_PATH}")
           dialog_layout = QVBoxLayout(plot_dialog)
           dialog_layout.addWidget(toolbar)
           dialog_layout.addWidget(canvas)
           plot_dialog.exec_()
           output_box.append("Data Loaded Successfully..!!") 
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    button_layout.itemAt(0).widget().clicked.connect(view_adc_data)     
    button_layout.itemAt(1).widget().clicked.connect(Perform_fft)
    button_layout.itemAt(2).widget().clicked.connect(view_noiseless_data)
    button_layout.itemAt(3).widget().clicked.connect(calculate_dist)
    button_layout.itemAt(4).widget().clicked.connect(view_peaks)
    
    dist_label = QLabel("0.0 cm")
    dist_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )

    button_layout.addWidget(dist_label)
    v2_layout.addLayout(button_layout)
    
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setLineWidth(2)
    v2_layout.addWidget(line)
    
    trainingmodelspacer_title = QLabel("Train Model")
    trainingmodelspacer_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    v2_layout.addWidget(trainingmodelspacer_title)
    
    Train_button_layout = QHBoxLayout()
    Train_button_layout.setSpacing(10)  # Spacing between buttons
    Train_button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
    Train_button_layout.setAlignment(Qt.AlignLeft)
    
    # Add new buttons
    select_folder_button = QPushButton("Select Folder")
    select_folder_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    select_folder_button.setFixedWidth(200)
    select_folder_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    Train_button_layout.addWidget(select_folder_button)
    
    folder_path_label = QLabel("No folder selected")
    folder_path_label.setStyleSheet(
        "font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;"
    )
    folder_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    Train_button_layout.addWidget(folder_path_label)
    
     # Function to open folder dialog and set path
    def select_folder():
        global FOLDER_PATH
        FOLDER_PATH = QFileDialog.getExistingDirectory(
            None, "Select Folder"
        )
        if FOLDER_PATH:
            folder_path_label.setText(FOLDER_PATH)
            output_box.append(f"Folder selected: {FOLDER_PATH}")
    
    select_folder_button.clicked.connect(select_folder)
    
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
    Train_button_layout.addWidget(progress_bar)
    
    def train_my_model():
        global FOLDER_PATH, model,X_test, y_test
        
        if FOLDER_PATH is None:
           output_box.append("No Folder is Loaded...!!")
           return
        try:
            output_box.append("ADC Signals Being Loaded...")
            adc_data_list = []
            lowcut = 39500.0  # Lower cutoff frequency in Hz
            highcut = 41500.0  # Upper cutoff frequency in Hz
            fs = 1.953125e6  # Sampling frequency in Hz (as per RED Pitaya)
            sos = processadcdata.butter_bandpass(lowcut, highcut, fs)
            QApplication.processEvents()
            
            files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt')]
            total_files = len(files)
            if total_files == 0:
                output_box.append("No valid files found in the folder.")
                return

            # Iterate over all files in the folder, label index starts from 1
            for idx, filename in enumerate(os.listdir(FOLDER_PATH), start=1):
                # Optionally, only process .txt files (adjust extension if needed)
                if filename.endswith('.txt'):
                    FILE_PATH = os.path.join(FOLDER_PATH, filename)
                    # Read ADC data from file using your get_adc_data function
                    my_adc_data = myadcdata.get_adc_data(FILE_PATH)
                    # Append a tuple containing the label (index) and the ADC data
                    adc_data_list.append((idx, my_adc_data)) 
                    progress = int((idx / total_files) * 28)
                    progress_bar.setValue(progress)
                    QApplication.processEvents()
                    
            output_box.append("ADC Signals Loaded...!!")
            output_box.append("Preparing Model to train...")
                   
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
                filter_progress = 28 + int((idx / total_files) * 15)
                progress_bar.setValue(filter_progress)
                QApplication.processEvents()
                    
            peak_features = train.process_adc_data_dataframe(filtered_myadc_data)
            df_features = train.consolidate_peak_features(peak_features)
            output_box.append("Data Prepared for Training...!!")
            
            distance_to_features = {}
    
            for file_label, filtered_df in filtered_data_list:
                # Step 1: Process the filtered ADC DataFrame to extract candidate peak features.
                peak_features = train.process_adc_data_dataframe(filtered_df)
                    
                # Step 2: Consolidate these features into a DataFrame.
                df_features = train.consolidate_peak_features(peak_features)
                    
                # Step 3: Detect the prominent peak from the filtered data.
                peak_index, peak_value = processadcdata.detect_prominent_peaks(filtered_df,std_multiplier=3)
                    
                # Step 4: Calculate the distance corresponding to the detected peak index.
                distance = processadcdata.calculate_distance_from_peak(peak_index)
                    
                # Step 5: Store the consolidated features in a dictionary, keyed by the calculated distance.
                distance_to_features[distance] = df_features
                filter_progress = 43 + int((idx / total_files) * 7)
                progress_bar.setValue(filter_progress)
                QApplication.processEvents()  
            
            dfs = []
            for dist, df in distance_to_features.items():
                df_copy = df.copy()  # avoid modifying the original DataFrame
                df_copy['distance'] = dist  # add the distance as a new column
                dfs.append(df_copy)
                QApplication.processEvents()
                
            output_box.append("Model ready for training...!!")
             
            combined_df = pd.concat(dfs, ignore_index=True)
            output_box.append("Training Started...!!")
            
            feature_columns = ['candidate_peak_index', 'amplitude', 'prominence', 'distance']
            X = combined_df[feature_columns]
            y = combined_df["label"]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a simple logistic regression model
            model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
            model.fit(X_train, y_train)
            progress_bar.setValue(73)
            # Predict on the test set
            y_pred = model.predict(X_test)
            output_box.append(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)}")
            output_box.append("\nClassification Report:")
            output_box.append(f"{classification_report(y_test, y_pred)}")
            progress_bar.setValue(85)
            # Set up 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Perform cross-validation using accuracy as the metric
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
            progress_bar.setValue(97)

            # Print the cross-validation scores and the mean accuracy
            output_box.append(f"Cross-Validation Accuracy Scores:{cv_scores}")
            output_box.append(f"Mean CV Accuracy:{np.mean(cv_scores)}")
            progress_bar.setValue(100)
            output_box.append("Training Completed!")
        except Exception as e:
            output_box.append(f"An error occurred while viewing data: {e}")
    
    def plot_model():
        global model, X_test, y_test
        
        if model is None:
            output_box.append("Error: No training model available. Train the model first.")
            return
        
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Create a figure to display the confusion matrix
            fig = plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
            plt.title("Confusion Matrix")
            plt.tight_layout()
            
            # Create a canvas and toolbar to embed the plot in a QDialog
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar2QT(canvas)
            
            plot_dialog = QDialog()
            plot_dialog.setWindowTitle("Model Performance")
            dialog_layout = QVBoxLayout(plot_dialog)
            dialog_layout.addWidget(toolbar)
            dialog_layout.addWidget(canvas)
            plot_dialog.exec_()      
            output_box.append("Model performance...")
        except Exception as e:
            output_box.append(f"Error plotting model performance: {e}")
       
    Train_button_layout.itemAt(2).widget().clicked.connect(train_my_model)
    Train_button_layout.itemAt(3).widget().clicked.connect(plot_model)
    
    save_model_button = QPushButton("Save Model")
    save_model_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    save_model_button.setFixedWidth(200)
    save_model_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    Train_button_layout.addWidget(save_model_button)
    
    def save_trained_model():
        global model
        if model is None:
            output_box.append("Error: No trained model found. Train the model first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(None, "Save Model", "model.pkl", "pkl Files (*.pkl);;All Files (*)")
        
        if file_path:  # If user selects a file path
           joblib.dump(model, file_path)
           output_box.append(f"Model saved successfully at: {file_path}")
    
    save_model_button.clicked.connect(save_trained_model)

    v2_layout.addLayout(Train_button_layout)
    
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setLineWidth(2)
    v2_layout.addWidget(line)
    
    predictmodelspacer_title = QLabel("Predict Model")
    predictmodelspacer_title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
    v2_layout.addWidget(predictmodelspacer_title)
    
    Predict_button_layout = QHBoxLayout()
    Predict_button_layout.setSpacing(10)  # Spacing between buttons
    Predict_button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
    Predict_button_layout.setAlignment(Qt.AlignLeft)
    
    # Add new buttons
    select_predictmodel_button = QPushButton("Select Model")
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
        global PREDICT_MODEL_PATH,Loadmodel
        PREDICT_MODEL_PATH, _ = QFileDialog.getOpenFileName(None, "Select Model", "", "H5 Files (*.pkl);;All Files (*)")
        if PREDICT_MODEL_PATH:
           
           Loadmodel = joblib.load(PREDICT_MODEL_PATH)
           model_path_label.setText(os.path.basename(PREDICT_MODEL_PATH))
           output_box.append(f"Model selected: {PREDICT_MODEL_PATH}")
        else:
           output_box.append("Invalid Model or File..!!")
           
    def select_predictfile():
        global PREDICT_MODEL_PATH,PREDICT_FILE_PATH
        PREDICT_FILE_PATH, _ = QFileDialog.getOpenFileName(None, "Select File", "", "Txt Files (*.txt);;All Files (*)")
        
        if "adc" not in PREDICT_FILE_PATH.lower():
            output_box.append("I think you have not selected ADC Data :( ")
            return
        else:
            predictfile_path_label.setText(os.path.basename(PREDICT_FILE_PATH))
            output_box.append("ADC Data Loaded Successfully.")
            try:
                output_box.append(f"File selected: {PREDICT_FILE_PATH}")
                
            except Exception as e:
                output_box.append(f"Error loading ADC data: {e}")
    
    def predict_adc():
        global PREDICT_MODEL_PATH,PREDICT_FILE_PATH,Loadmodel
        if "adc" not in PREDICT_FILE_PATH.lower():
            output_box.append("I think you have not selected ADC Data :( ")
            return
        elif Loadmodel is None:
            output_box.append("Error: No trained model found. Train the model first.")
            return
        elif PREDICT_FILE_PATH is not None :
            try:
                predictdata = PredictADC()
                lowcut = 39500.0  # Lower cutoff frequency in Hz
                highcut = 41500.0  # Upper cutoff frequency in Hz
                fs = 1.953125e6  # Sampling frequency in Hz (as per RED Pitaya)
                sos = processadcdata.butter_bandpass(lowcut, highcut, fs)
                
                predict_adc_data = myadcdata.get_adc_data(PREDICT_FILE_PATH)
                
                adc_data_array = predict_adc_data.to_numpy()
                filtered_data_array = np.apply_along_axis(
                    processadcdata.apply_bandpass_filter, 1, adc_data_array, sos
                )
                filtered_myadc_data = pd.DataFrame(filtered_data_array)
                
                peak_features = train.process_adc_data_dataframe(filtered_myadc_data)
                df_features = train.consolidate_peak_features(peak_features)
                
                df_features = df_features.copy()
                df_features["distance"] = 0.0
                feature_columns = ['candidate_peak_index', 'amplitude', 'prominence', 'distance']   
                df_features["predicted_label"] = Loadmodel.predict(df_features[feature_columns])
                
                df_first_echo = predictdata.identify_first_echo(df_features)

                predicted_peaks = df_features[df_features["predicted_label"] == 1].copy()

                # Check if any predicted peaks exist
                if not predicted_peaks.empty:
                    # Find the row with the maximum amplitude among predicted peaks
                    max_peak_row = predicted_peaks.loc[predicted_peaks["amplitude"].idxmax()]

                    highest_peak_index = max_peak_row["candidate_peak_index"]
                    highest_peak_amplitude = max_peak_row["amplitude"]

                    output_box.append(f"Highest Predicted Peak Index: {highest_peak_index}")
                    output_box.append(f"Highest Predicted Peak Amplitude: {highest_peak_amplitude}")
                    
                    dist = (processadcdata.calculate_distance_from_peak(highest_peak_index))*100
                    output_box.append(f"First Echo Detected at Dist: {dist} cm")
                else:
                    output_box.append("No predicted peaks were found.")
                    return
                
                fig = plt.figure(figsize=(10, 6))
                # Plot all filtered ADC signal rows in light gray.
                for index, row in filtered_myadc_data.iterrows():
                    plt.plot(row, color="lightgray", linewidth=0.5, alpha=0.7)
                
                # Mark each predicted peak with an "x"
                for idx, row in predicted_peaks.iterrows():
                    x_val = row["candidate_peak_index"]
                    y_val = row["amplitude"]
                    plt.plot(x_val, y_val, "x", markersize=5)
                
                # Draw vertical line for highest predicted peak index
                plt.axvline(x=highest_peak_index, color="red", linestyle="--", 
                            label=f"Peak Index: {highest_peak_index}")
                # Draw horizontal line for highest predicted peak amplitude
                plt.axhline(y=highest_peak_amplitude, color="blue", linestyle="--", 
                            label=f"Peak Amplitude: {highest_peak_amplitude}")
                
                plt.title("Filtered ADC Signals with Predicted Peaks")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar2QT(canvas)
                plot_dialog = QDialog()
                plot_dialog.setWindowTitle(f"{PREDICT_FILE_PATH}")
                dialog_layout = QVBoxLayout(plot_dialog)
                dialog_layout.addWidget(toolbar)
                dialog_layout.addWidget(canvas)
                plot_dialog.exec_()
                output_box.append("Data Loaded Successfully..!!")
                PREDICT_FILE_PATH = None
                
            except Exception as e:
                output_box.append(f"Error loading ADC data: {e}")
                    
    select_predictmodel_button.clicked.connect(select_predictionmodel)
    select_predictfile_button.clicked.connect(select_predictfile)
    predictselectedfile_button.clicked.connect(predict_adc)
            
    v2_layout.addLayout(Predict_button_layout)
    
    version2_widget.setLayout(v2_layout)

    # Add the dummy widgets to the stacked widget
    stacked_widget.addWidget(version1_widget)  # index 0 for Version 1
    stacked_widget.addWidget(version2_widget)  # index 1 for Version 2

    # Add the stacked widget to the main layout
    layout.addWidget(stacked_widget)

    # Connect the toggled signal of each radio button to update the stacked widget and log the selection
    def select_version1(checked):
        if checked:
            stacked_widget.setCurrentIndex(0)
            output_box.append("Version 1 selected.")

    def select_version2(checked):
        if checked:
            stacked_widget.setCurrentIndex(1)
            output_box.append("Version 2 selected.")

    radio_v1.toggled.connect(select_version1)
    radio_v2.toggled.connect(select_version2)
