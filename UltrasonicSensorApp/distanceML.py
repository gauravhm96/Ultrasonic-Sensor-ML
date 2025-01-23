import numpy as np
import time
import os
from PyQt5.QtWidgets import (QPushButton, QLabel, QVBoxLayout,
                             QFileDialog,QSizePolicy,QHBoxLayout,
                             QDialog,QProgressBar,QApplication,QSpacerItem,QTextEdit,QGridLayout)
from PyQt5.QtCore import Qt,QCoreApplication

from SignalProcess import SignalProcessor
from ObjectDetectionFeatureExtract import FeatureExtract

def distanceMLFeatureExtract(layout,output_box):
    
    extractfeature = FeatureExtract()
    ProcessSignal = SignalProcessor()
    shared_data = {}
    
    feature_label = QLabel("Feature Extraction, Training & Predict.")
    feature_label.setStyleSheet("font-size: 18px; font-weight: normal;padding: 5px")
    layout.addWidget(feature_label)
    
    file_layout = QHBoxLayout()
    
    select_file_button  = QPushButton("Select File")
    select_file_button.setStyleSheet("font-size: 18px; padding: 5px;")
    select_file_button.setFixedWidth(select_file_button.sizeHint().width())
    
    # Add button to layout
    file_layout.addWidget(select_file_button)
    
    # Add a QLabel to show the file path
    file_path_label = QLabel("No file selected")
    file_path_label.setStyleSheet("font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;")
    file_layout.addWidget(file_path_label)

    layout.addLayout(file_layout)
    
    # Function to open file dialog and set path
    def select_file():
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Signal File", "", "Text Files (*.txt);;CSV Files (*.csv)")
        if file_path:
            # Set the path in the label
            file_path_label.setText(file_path)
            # Print path in the output box
            output_box.append(f"File selected: {file_path}")
            ProcessSignal.set_file_path(file_path)
            ProcessSignal.load_signal_data()
            Analyze_button.setEnabled(True)
    
    # Connect the button to the select file function
    select_file_button.clicked.connect(select_file)
    
    analyze_layout = QHBoxLayout()
    
    # Adding the "Calculate Distance" button
    Analyze_button = QPushButton("Analyze Signal")
    Analyze_button.setStyleSheet("font-size: 18px; padding: 5px;")
    Analyze_button.setFixedWidth(Analyze_button.sizeHint().width())
    Analyze_button.setEnabled(False)

    analyze_layout.addWidget(Analyze_button)
        
    progress_bar = QProgressBar()
    progress_bar.setStyleSheet("font-size: 18px;")
    progress_bar.setValue(0)  
    progress_bar.setFixedHeight(Analyze_button.sizeHint().height())
    progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    analyze_layout.addWidget(progress_bar,stretch=1)    
    
    layout.addLayout(analyze_layout)
    
    def analyze_signal():
        progress_bar.setValue(0)
        output_box.append("Signal analysis started...")
        
        def progress(start, end):
            for value in range(start, end + 1):
                QCoreApplication.processEvents()  # Keep GUI responsive
                progress_bar.setValue(value)
                time.sleep(0.05)  
        
        ProcessSignal.load_signal_data()
        progress(0, 20)  # Update progress to 20%
        output_box.append("Signal data loaded.")
        
        ProcessSignal.analyze_raw_signals()
        progress(20, 40)  # Update progress to 40%
        output_box.append("Raw signals analyzed.")
        
        ProcessSignal.annotate_real_peaks()
        progress(40, 50)  # Update progress to 50%
        output_box.append("Real peaks annotated.")
        
        updated_signals, threshold_info = ProcessSignal.NoiseFiltering()
        progress(50, 60)  # Update progress to 60%
        output_box.append("Noise filtering applied.")        
        output_box.append(f"Updated Signals: {updated_signals}")
        output_box.append(f"Threshold Info: {threshold_info}")
        
        updated_signals, overall_threshold = extractfeature.apply_threshold_filtering(updated_signals)
        progress(60, 70)  # Update progress to 70%
        output_box.append("Threshold filtering applied.")
        output_box.append(f"Updated Signals: {updated_signals}")
        output_box.append(f"Threshold Info: {overall_threshold}")
        shared_data['updated_signals'] = updated_signals
        
        selected_peak_windows = extractfeature.extract_peak_windows(updated_signals)
        progress(70, 80)  # Update progress to 80%
        output_box.append("Peak windows extracted.")
        output_box.append(f"Updated Signals: {selected_peak_windows}")
        shared_data['selected_peak_windows'] = selected_peak_windows
        
        
        selected_non_peak_windows = extractfeature.extract_non_peak_windows(updated_signals)
        progress(80, 90) # Update progress to 90%
        output_box.append(f"Updated Signals: {selected_non_peak_windows}")
        shared_data['selected_non_peak_windows'] = selected_non_peak_windows        
        
        window_duration,ADC_SAMPLE_FREQUENCY = extractfeature.calulate_window(num_samples_per_window=300)
        progress(90, 95)  # Update progress to 90%
        output_box.append("window_duration Determined!")      
        output_box.append(f"Window duration: {window_duration}, ADC Sample Frequency: {ADC_SAMPLE_FREQUENCY}")
        shared_data['ADC_SAMPLE_FREQUENCY'] = ADC_SAMPLE_FREQUENCY
        
        output_box.append("Feature Extraction Completed..!!")
        progress_bar.setValue(100)  # Update progress to 100%
        GenerateSpectogramPeaks_button.setEnabled(True)
        GenerateSpectogramPeaksType2_button.setEnabled(True)
        GenerateSpectogramNonPeaks_button.setEnabled(True)
        
    Analyze_button.clicked.connect(analyze_signal)
    
    PeakSpectogramLayout = QHBoxLayout()
    
    GenerateSpectogramPeaks_button = QPushButton("Generate Spectrograms Peaks(Type1)")
    GenerateSpectogramPeaks_button.setStyleSheet("font-size: 18px; padding: 5px;")
    GenerateSpectogramPeaks_button.setFixedWidth(GenerateSpectogramPeaks_button.sizeHint().width())
    GenerateSpectogramPeaks_button.setEnabled(False)
    

    PeakSpectogramLayout.addWidget(GenerateSpectogramPeaks_button)
        
    Peaksprogress_bar = QProgressBar()
    Peaksprogress_bar.setStyleSheet("font-size: 18px;")
    Peaksprogress_bar.setValue(0)  
    Peaksprogress_bar.setFixedHeight(40)
    Peaksprogress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    PeakSpectogramLayout.addWidget(Peaksprogress_bar)    
    
    layout.addLayout(PeakSpectogramLayout)
    
    def Generate_Spectogram_PeaksType1():
        
        Peaksprogress_bar.setValue(0)
        output_box.append("Generating Spectrograms...")
               
        folder_path = os.path.dirname(os.path.abspath(__file__))
        PeakSpectrogramType1 = os.path.join(folder_path, "PeakspectrogramType1")
        
        if not os.path.exists(PeakSpectrogramType1):
            os.makedirs(PeakSpectrogramType1)
            output_box.append(f"Folder 'Peakspectrogram' created at {PeakSpectrogramType1}")
        else:
            output_box.append(f"Folder 'Peakspectrogram' already exists at {PeakSpectrogramType1}")
        
        
        output_box.append("Preparing to save spectrograms...")
        Peaksprogress_bar.setValue(10)
        
        output_box.append("Processing selected peaks...")
        
        Peaksprogress_bar.setValue(30)
        QCoreApplication.processEvents() 
        
        extractfeature.save_PeakSspectrogramsType_1(shared_data['selected_peak_windows'],
                                                    shared_data['updated_signals'],
                                                    PeakSpectrogramType1,
                                                    num_samples_per_window= 300,ADC_SAMPLE_FREQUENCY=shared_data["ADC_SAMPLE_FREQUENCY"]
                                                    )
        
        output_box.append("Saving spectrograms...")
        Peaksprogress_bar.setValue(100)
        
        output_box.append("Spectrograms Generated...!!")
        
    GenerateSpectogramPeaks_button.clicked.connect(Generate_Spectogram_PeaksType1)
    
    PeakSpectogramtype2Layout = QHBoxLayout()
    
    GenerateSpectogramPeaksType2_button = QPushButton("Generate Spectrograms Peaks(Type2)")
    GenerateSpectogramPeaksType2_button.setStyleSheet("font-size: 18px; padding: 5px;")
    GenerateSpectogramPeaksType2_button.setFixedWidth(GenerateSpectogramPeaksType2_button.sizeHint().width())
    GenerateSpectogramPeaksType2_button.setEnabled(False)
    
    PeakSpectogramtype2Layout.addWidget(GenerateSpectogramPeaksType2_button)
        
    Peakstype2progress_bar = QProgressBar()
    Peakstype2progress_bar.setStyleSheet("font-size: 18px;")
    Peakstype2progress_bar.setValue(0)  
    Peakstype2progress_bar.setFixedHeight(40)
    Peakstype2progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    PeakSpectogramtype2Layout.addWidget(Peakstype2progress_bar)    
    
    layout.addLayout(PeakSpectogramtype2Layout)

    def Generate_Spectogram_PeaksType2():
        
        Peakstype2progress_bar.setValue(0)
        output_box.append("Generating Spectrograms...")
               
        folder_path = os.path.dirname(os.path.abspath(__file__))
        PeakSpectrogramType2 = os.path.join(folder_path, "PeakspectrogramType2")
        
        if not os.path.exists(PeakSpectrogramType2):
            os.makedirs(PeakSpectrogramType2)
            output_box.append(f"Folder 'Peakspectrogram' created at {PeakSpectrogramType2}")
        else:
            output_box.append(f"Folder 'Peakspectrogram' already exists at {PeakSpectrogramType2}")
        
        output_box.append("Preparing to save spectrograms...")
        Peakstype2progress_bar.setValue(10)
        
        output_box.append("Processing selected peaks...")
        
        Peakstype2progress_bar.setValue(30)
        QCoreApplication.processEvents() 
               
        extractfeature.save_PeakSspectrogramsType_2(shared_data['selected_peak_windows'], PeakSpectrogramType2, figure_size = (3, 3))
        
        output_box.append("Saving spectrograms...")
        Peakstype2progress_bar.setValue(100)
        
        output_box.append("Spectrograms Generated...!!")
        
    GenerateSpectogramPeaksType2_button.clicked.connect(Generate_Spectogram_PeaksType2)
    
    NonPeakSpectogramLayout = QHBoxLayout()
    
    GenerateSpectogramNonPeaks_button = QPushButton("Generate Spectrograms Non Peaks")
    GenerateSpectogramNonPeaks_button.setStyleSheet("font-size: 18px; padding: 5px;")
    GenerateSpectogramNonPeaks_button.setFixedWidth(GenerateSpectogramNonPeaks_button.sizeHint().width())
    GenerateSpectogramNonPeaks_button.setEnabled(False)
    

    NonPeakSpectogramLayout.addWidget(GenerateSpectogramNonPeaks_button)
        
    NonPeaksprogress_bar = QProgressBar()
    NonPeaksprogress_bar.setStyleSheet("font-size: 18px;")
    NonPeaksprogress_bar.setValue(0)  
    NonPeaksprogress_bar.setFixedHeight(40)
    NonPeaksprogress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    
    NonPeakSpectogramLayout.addWidget(NonPeaksprogress_bar)    
    
    layout.addLayout(NonPeakSpectogramLayout)
    
    def Generate_Spectogram_NonPeaks():
        NonPeaksprogress_bar.setValue(0)
        output_box.append("Generating Spectrograms...")
               
        folder_path = os.path.dirname(os.path.abspath(__file__))
        NonPeakSpectrogram = os.path.join(folder_path, "NonPeakspectrogram")
        
        if not os.path.exists(NonPeakSpectrogram):
            os.makedirs(NonPeakSpectrogram)
            output_box.append(f"Folder 'Peakspectrogram' created at {NonPeakSpectrogram}")
        else:
            output_box.append(f"Folder 'Peakspectrogram' already exists at {NonPeakSpectrogram}")
        
        output_box.append("Preparing to save spectrograms...")
        NonPeaksprogress_bar.setValue(10)
        
        output_box.append("Processing selected peaks...")
        
        NonPeaksprogress_bar.setValue(30)
        QCoreApplication.processEvents() 
               
        extractfeature.save_NonPeakSspectrograms(shared_data['selected_non_peak_windows'], NonPeakSpectrogram)
        
        output_box.append("Saving spectrograms...")
        NonPeaksprogress_bar.setValue(100)
        
        output_box.append("Spectrograms Generated...!!")
        
    GenerateSpectogramNonPeaks_button.clicked.connect(Generate_Spectogram_NonPeaks)
    