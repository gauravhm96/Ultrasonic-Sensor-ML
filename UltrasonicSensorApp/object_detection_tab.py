from PyQt5.QtWidgets import QLabel, QHBoxLayout, QPushButton, QFileDialog,QSizePolicy,QDialog,QSpacerItem,QProgressBar,QApplication,QTextEdit,QGridLayout
from FftSignalProcess import FftSignal,getFFTSignalParameters,getFFTSignalFreqDomainFeatures
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtWidgets import QDialog, QVBoxLayout
import pandas as pd
from PyQt5.QtCore import Qt
import numpy as np
import time

from trainfftsignal import FFTModel
from Predictfft import Predict_FFT
import os
from SignalProcess import SignalProcessor
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

def object_detection_features(layout, output_box):
    signalprocess = SignalProcessor()
    
    feature_label = QLabel("Computation for ADC Data Acquired from Ultrasonic Sensor for Object Detection & Distance Estimation")
    feature_label.setStyleSheet("font-size: 18px; font-weight: normal;padding: 5px")
    layout.addWidget(feature_label)

    file_layout = QHBoxLayout()
    file_layout.setSpacing(10)      
    file_layout.setContentsMargins(0, 0, 0, 0)      
    file_layout.setAlignment(Qt.AlignLeft)
    
    select_file_button = QPushButton("Select ADC Signal")
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
            None, "Select ADC File", "", "ADC Data (*.txt);;CSV (*.csv)"
        )
        if file_path:
            if "adc" not in file_path.lower():
                output_box.append("I think you have not selected ADC Data :( ")
                return
            else :
                file_path_label.setText(file_path)
                output_box.append(f"File selected: {file_path}")
                try:
                    signalprocess.set_file_path(file_path)
                    signal_data = signalprocess.load_signal_data()
                    output_box.append("ADC Data Loaded Successfully.\n"
                                  f"Total Data Points: {len(signal_data)}\n"
                                  "Processing data...\n")
                    output_box.append(f"ADC Signal:\n{signal_data[:10]}")
                except Exception as e:
                    output_box.append(f"Error loading ADC data: {e}")
                    
    # Connect the button to the select file function
    select_file_button.clicked.connect(select_file)
    
    calculatebutton_layout = QHBoxLayout()
    calculatebutton_layout.setSpacing(10)   
    calculatebutton_layout.setContentsMargins(0, 0, 0, 0)   
    calculatebutton_layout.setAlignment(Qt.AlignLeft)
    
    calculatedist_button = QPushButton("Calculate Distance")
    calculatedist_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    calculatedist_button.setFixedWidth(200)
    calculatedist_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    calculatebutton_layout.addWidget(calculatedist_button)
    
    # Distance Print Label
    calculatedist_label = QLabel("0.0")
    calculatedist_label.setStyleSheet("font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;")
    calculatedist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    calculatebutton_layout.addWidget(calculatedist_label)
    
    show_computation_button = QPushButton("Show Computation")
    show_computation_button.setStyleSheet("font-size: 18px;font-weight: normal; padding: 5px;")
    show_computation_button.setFixedWidth(200)
    calculatebutton_layout.addWidget(show_computation_button)

    def calculate_distance():
        dialog = QDialog()
        dialog.setWindowTitle("Processing....")
        dialog.setFixedSize(500, 150)
        
        layout = QVBoxLayout()
        progress_label = QLabel("Calculating distance....")
        progress_label.setStyleSheet("font-size: 16px; padding: 5px;")
        progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(progress_label)
        
        progress_bar = QProgressBar()
        progress_bar.setStyleSheet("""QProgressBar {font-size: 16px;padding: 5px;text-align: center;}
                                      QProgressBar::chunk {background-color: #4caf50;width: 1px;}""")
        progress_bar.setRange(0, 100)
        layout.addWidget(progress_bar)
        
        # Ok button
        ok_button = QPushButton("Ok")
        ok_button.setEnabled(False)  # Initially disabled
        ok_button.setFixedSize(80, 30)
        ok_button.setStyleSheet("font-size: 14px; padding: 5px;")
        ok_button.clicked.connect(dialog.close)
        layout.addWidget(ok_button,alignment=Qt.AlignCenter)
        
        layout.setAlignment(Qt.AlignCenter)
        dialog.setLayout(layout)
        dialog.show()
        
        # Simulate a long process by updating the progress bar in increments
        for i in range(101):  # Increment from 0 to 100
            progress_bar.setValue(i)
            time.sleep(0.05)  # Simulate a time-consuming task
            QApplication.processEvents()  # Update the UI during the loop

        if signalprocess:
            try:
                # Process the signal
                signalprocess.analyze_raw_signals()
                signalprocess.annotate_real_peaks()
                signalprocess.NoiseFiltering()
                signalprocess.SignalCorrection()

                # Calculate the distance
                distance_info = signalprocess.Calculate_Distance()
                
                 # Extract the distance value from the returned dictionary
                distance = distance_info.get("distance", 0.0)

                # Update the label with the calculated distance
                calculatedist_label.setText(f"{distance:.2f} m")
                output_box.append("Distance calculation completed successfully.")
                progress_label.setText(f"Calculation completed! Distance: {distance:.2f} m")
            except Exception as e:
                output_box.append(f"Error during distance calculation: {e}")
                progress_label.setText("Signal processor not initialized.")
        else:
            output_box.append("Signal processor not initialized.")
        
        ok_button.setEnabled(True)
        dialog.exec()
    
    def show_calculation():
        output_box.append("Computation results will be displayed here.")
    
    calculatedist_button.clicked.connect(calculate_distance)
    show_computation_button.clicked.connect(show_calculation)

    layout.addLayout(calculatebutton_layout)

    