from PyQt5.QtWidgets import (QPushButton, QLabel, QVBoxLayout, 
                             QFileDialog,QSizePolicy,QHBoxLayout,
                             QDialog,QProgressBar,QApplication,QSpacerItem,QTextEdit,QGridLayout)
from PyQt5.QtCore import Qt
from SignalProcess import SignalProcessor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import numpy as np

def add_general_features(layout,output_box):
   
    signalprocess = SignalProcessor()

    feature_label = QLabel("This is a feature label for the general tab.")
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
            signalprocess.set_file_path(file_path)
            signalprocess.load_signal_data()

    # Connect the button to the select file function
    select_file_button.clicked.connect(select_file)

    # Adding the "Calculate Distance" button
    calculate_distance_button = QPushButton("Calculate Distance")
    calculate_distance_button.setStyleSheet("font-size: 18px; padding: 5px;")
    calculate_distance_button.setFixedWidth(calculate_distance_button.sizeHint().width())
    
    layout.addWidget(calculate_distance_button)

    # Add a QLabel to show the file path
    distance_label = QHBoxLayout()
    distance_text_label = QLabel("Distance from the Sensor to the Object: ")
    distance_text_label.setStyleSheet("font-size: 18px; padding: 5px;")
    distance_label.addWidget(distance_text_label)

    calculate_distance_label = QLabel("0.00")
    calculate_distance_label.setStyleSheet("font-size: 18px; padding: 5px; border: 1px solid black;background-color: white;")
    distance_label.addWidget(calculate_distance_label)

    distance_label.addStretch()
    layout.addLayout(distance_label)
    
    # Define a function to calculate distance
    def calculate_distance():
        dialog = QDialog()
        dialog.setWindowTitle("Processing")
        dialog.setFixedSize(500, 150)
        
        layout = QVBoxLayout()
        progress_label = QLabel("Calculating distance")
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
                calculate_distance_label.setText(f"{distance:.2f} m")
                output_box.append("Distance calculation completed successfully.")
                progress_label.setText(f"Calculation completed! Distance: {distance:.2f} m")
            except Exception as e:
                output_box.append(f"Error during distance calculation: {e}")
                progress_label.setText("Signal processor not initialized.")
        else:
            output_box.append("Signal processor not initialized.")
        
        ok_button.setEnabled(True)
        dialog.exec()
        
        
    calculate_distance_button.clicked.connect(calculate_distance)

    # Adding the "Show Computation" button
    show_computation_button = QPushButton("Show Calculations")
    show_computation_button.setStyleSheet("font-size: 18px; padding: 5px;")
    show_computation_button.setFixedWidth(show_computation_button.sizeHint().width())
    layout.addWidget(show_computation_button)

    def show_calculation():
       
        dialog = QDialog()
        dialog.setWindowTitle("Computation Results")
        dialog.setFixedSize(2400, 1250)
        
        # Set up the layout for the dialog
        layout = QVBoxLayout()
        
         # Add a title label
        title_label = QLabel("Calculations")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Add a text area to display results
        results_label = QLabel("Distance calculation and signal processing details.")
        results_label.setStyleSheet("font-size: 18px; padding: 10px;")
        results_label.setWordWrap(True)
        layout.addWidget(results_label)
        
        # Add a "Read Data" button
        read_data_button = QPushButton("Read Data")
        
        read_data_button.setFixedSize(100, 30)
        read_data_button.setFixedWidth(show_computation_button.sizeHint().width())
        read_data_button.setStyleSheet("font-size: 18px; padding: 5px;")
        layout.addWidget(read_data_button, alignment=Qt.AlignLeft)
        
            # Add an output box below the "Read Data" button
        output_box = QTextEdit()
        output_box.setReadOnly(True) 
        output_box.setStyleSheet("font-size: 16px; padding: 10px; background-color: #f4f4f4; border: 1px solid #ccc;")
        output_box.setFixedHeight(200)  # Set a fixed height for the output box
        layout.addWidget(output_box)
        
        # Create a grid layout for the plots
        plots_layout = QGridLayout()
        # Add 8 plots in a 2x4 grid
        
        fig1, axs1 = signalprocess.PlotRawSignal()
        fig1.set_size_inches(10, 6)
        canvas1 = FigureCanvas(fig1)
        
        fig2,axs2 = signalprocess.PlotNoiseFilteredSignal()
        fig2.set_size_inches(10, 6)
        canvas2 = FigureCanvas(fig2)
        
        fig3,axs3 = signalprocess.PlotSignalCorrection()
        fig3.set_size_inches(20, 6)
        canvas3 = FigureCanvas(fig3)

        plots_layout.setSpacing(20)
        plots_layout.setContentsMargins(20, 20, 20, 20) 
        plots_layout.addWidget(canvas1, 0, 0)
        plots_layout.addWidget(canvas2, 0, 1)
        plots_layout.addWidget(canvas3, 1, 0, 1, 2)
        
        # Add the plots layout to the main layout
        layout.addLayout(plots_layout)
        
        # Connect the "Read Data" button to a function
        def read_data():
            try:
                input_data = signalprocess.PrintInputdata()        
                output_box.append(f"Input Date:\n{input_data}")
            except Exception as e:
                output_box.append(f"Error Reading Data:{e}")

        read_data_button.clicked.connect(read_data)

        # Add a spacer to push the Close button to the bottom
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        
        # Add a close button
        close_button = QPushButton("Close")
        close_button.setFixedSize(100, 30)
        close_button.setStyleSheet("font-size: 18px; padding: 5px;")
        close_button.clicked.connect(dialog.close)  # Close the dialog when clicked
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        dialog.setLayout(layout)
        dialog.exec()
       
        output_box.append("Computation results will be displayed here.")

    show_computation_button.clicked.connect(show_calculation)
