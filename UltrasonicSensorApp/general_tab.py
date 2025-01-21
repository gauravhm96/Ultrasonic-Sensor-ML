from PyQt5.QtWidgets import QPushButton, QLabel, QVBoxLayout, QFileDialog,QSizePolicy,QHBoxLayout,QDialog,QProgressBar,QApplication
from PyQt5.QtCore import Qt
from SignalProcess import SignalProcessor
import time

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
    distance_text_label = QLabel("Distance: ")
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
            time.sleep(0.03)  # Simulate a time-consuming task
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
    show_computation_button = QPushButton("Show Computation")
    show_computation_button.setStyleSheet("font-size: 18px; padding: 5px;")
    show_computation_button.setFixedWidth(show_computation_button.sizeHint().width())
    layout.addWidget(show_computation_button)

    def show_computation():
       # Placeholder: Add the logic to show computation results
       output_box.append("Computation results will be displayed here.")

    show_computation_button.clicked.connect(show_computation)
