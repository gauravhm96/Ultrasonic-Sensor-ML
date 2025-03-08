import sys
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QSizePolicy, QSpacerItem, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)

from object_detection_tab import object_detection_features
from object_differentiation_tab import object_differentiation_features
from ultrasonic_red_pitaya_connect import connect_to_red_pitaya

class SensorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrasonic Sensor Application")
        self.setMinimumSize(1200, 1500)

        # Create a central widget and a vertical layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()

        # Add a logo at the top
        logo_label = QLabel()
        cwd = os.getcwd()
        logo_path = os.path.join(cwd, "Logo", "image(1).png")
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            print("Logo loaded successfully.")
        else:
            print("Failed to load logo. Check file path or format.")
        logo_label.setPixmap(logo_pixmap)
        logo_label.setScaledContents(True)
        logo_label.setFixedSize(300, 120)  # Adjust size as needed
        logo_label.setStyleSheet("margin: 10px;")
        logo_label.setAlignment(Qt.AlignCenter)

        # Create a tab widget
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("QTabBar::tab { padding: 10px; }")

        # Create a text output dialog box for logs
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)  # Make it read-only
        self.output_box.setStyleSheet(
            "background-color: white; border: 1px solid gray; font-size: 18px; padding: 5px;"
        )
        self.output_box.setFixedHeight(250)  

        # Add tabs to the tab widget
        tab_widget.addTab(self.Object_Differentiation(), "Object Differentiation")
        tab_widget.addTab(self.Object_Detection(), "Object Detection")
        tab_widget.addTab(self.advanced_tab(), "Connect To Red Pitaya")
        tab_widget.addTab(self.about_tab(), "About")
        # tab_widget.addTab(self.ML_tab(), "ML Tab")

        # Add OK, Cancel, and Exit buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setSpacing(20)

        clearlogs_button = QPushButton("Clear Logs")
        cancel_button = QPushButton("Cancel")
        exit_button = QPushButton("Exit")

        # Set styles and size for buttons
        for button in [clearlogs_button, cancel_button, exit_button]:
            button.setFixedSize(100, 40)

        # Add functionality to buttons
        clearlogs_button.clicked.connect(self.clearlogs_action)
        cancel_button.clicked.connect(self.cancel_action)
        exit_button.clicked.connect(self.close)

        # Add buttons to the layout with spacers
        button_layout.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        button_layout.addWidget(clearlogs_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(exit_button)

        # Add the logo, tab widget, output box, and buttons to the central layout
        central_layout.addWidget(logo_label)
        central_layout.addWidget(tab_widget)
        central_layout.addWidget(self.output_box)
        central_layout.addLayout(button_layout)

        # Set the layout to the central widget
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    def Object_Differentiation(self):
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Object Differentiation")
        label.setStyleSheet("font-size: 18px; font-weight: bold;padding: 5px;")

        layout.addWidget(label)

        object_differentiation_features(layout, self.output_box)

        layout.addStretch()  # Push contents to the top
        tab.setLayout(layout)
        return tab

    def Object_Detection(self):
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Object Detection")
        label.setStyleSheet("font-size: 18px; font-weight: bold;padding: 5px;")

        layout.addWidget(label)

        object_detection_features(layout, self.output_box)

        layout.addStretch()  # Push contents to the top
        tab.setLayout(layout)
        return tab
    
    def advanced_tab(self):
        """Create the Advanced tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Advanced Operations")
        label.setStyleSheet("font-size: 18px; font-weight: bold;padding: 5px;")

        layout.addWidget(label)
        
        connect_to_red_pitaya(layout, self.output_box)

        layout.addStretch()   
        tab.setLayout(layout)
        return tab
        
    def about_tab(self):
        """Create the About tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("About Ultrasonic App")
        label.setStyleSheet("font-size: 18px; font-weight: bold;padding: 5px;")

        description = QLabel(
            "This is a sample settings application created using PyQt5.\n\nVersion: 1.0"
        )
        description.setStyleSheet("font-size: 18px; font-weight: normal;padding: 5px")
        
        description.setWordWrap(True)

        layout.addWidget(label)
        layout.addWidget(description)
        layout.addStretch()  # Push contents to the top
        tab.setLayout(layout)
        return tab
    
    def clearlogs_action(self):
        self.output_box.clear()

    def cancel_action(self):
        """Handle Cancel button click."""
        sys.exit("Program terminated by user.")
