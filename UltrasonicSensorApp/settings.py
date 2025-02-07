import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from distanceML import distanceMLFeatureExtract
from general_tab import add_general_features
from object_differentiation_tab import object_differentiation_features


class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setMinimumSize(1500, 1000)

        # Create a central widget and a vertical layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()

        # Add a logo at the top
        logo_label = QLabel()
        logo_path = r"C:\@DevDocs\Projects\Mine\New folder\Ultrasonic-Sensor-ML\UltrasonicSensorApp\Logo\image(1).png"
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
        self.output_box.setFixedHeight(250)  # Adjust height as needed

        # Add tabs to the tab widget
        tab_widget.addTab(self.general_tab(), "General")
        tab_widget.addTab(self.ML_tab(), "ML Tab")
        tab_widget.addTab(self.advanced_tab(), "Advanced")
        tab_widget.addTab(self.Object_Differentiation(), "Object Differentiation")
        tab_widget.addTab(self.about_tab(), "About")

        # Add OK, Cancel, and Exit buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.setSpacing(20)

        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        exit_button = QPushButton("Exit")

        # Set styles and size for buttons
        for button in [ok_button, cancel_button, exit_button]:
            button.setFixedSize(100, 40)

        # Add functionality to buttons
        ok_button.clicked.connect(self.ok_action)
        cancel_button.clicked.connect(self.cancel_action)
        exit_button.clicked.connect(self.close)

        # Add buttons to the layout with spacers
        button_layout.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        button_layout.addWidget(ok_button)
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

    def ok_action(self):
        """Handle OK button click."""
        self.output_box.append("OK button clicked.")
        print("OK button clicked.")

    def cancel_action(self):
        """Handle Cancel button click."""
        sys.exit("Program terminated by user.")

    def general_tab(self):

        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("General Settings")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(label)

        # Call the function from features.py to add more features
        add_general_features(
            layout, self.output_box
        )  # This will add the additional features to the layout

        layout.addStretch()  # Push contents to the top

        tab.setLayout(layout)
        return tab

    def ML_tab(self):

        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Machine Learning")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(label)

        # Call the function from features.py to add more features
        distanceMLFeatureExtract(layout, self.output_box)

        layout.addStretch()  # Push contents to the top

        tab.setLayout(layout)
        return tab

    def advanced_tab(self):
        """Create the Advanced tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Advanced Settings")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")

        example_button = QPushButton("Configure Environment Variables")
        example_button.setStyleSheet(
            "background-color: lightgray; padding: 10px; font-size: 14px;"
        )

        layout.addWidget(label)
        layout.addWidget(example_button)
        layout.addStretch()  # Push contents to the top
        tab.setLayout(layout)
        return tab
    
    def Object_Differentiation(self):
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("Object Differentiation")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(label)
        
        object_differentiation_features(layout, self.output_box)
        
        layout.addStretch()  # Push contents to the top
        tab.setLayout(layout)
        return tab

    def about_tab(self):
        """Create the About tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        label = QLabel("About This Application")
        label.setStyleSheet("font-size: 18px; font-weight: bold;")

        description = QLabel(
            "This is a sample settings application created using PyQt5.\n\nVersion: 1.0"
        )
        description.setStyleSheet("font-size: 14px;")
        description.setWordWrap(True)

        layout.addWidget(label)
        layout.addWidget(description)
        layout.addStretch()  # Push contents to the top
        tab.setLayout(layout)
        return tab
    



