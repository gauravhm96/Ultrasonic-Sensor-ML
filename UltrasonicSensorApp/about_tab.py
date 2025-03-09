from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel,QFrame,QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import os

def about_tab(layout, output_box):

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setLineWidth(2)
    layout.addWidget(line)

    description1 = QLabel(
        "<b>Key Functionalities:</b><br>"
        "<ul style='margin: 0; padding-left: 20px;'>"
        "<li>Carry out Complex Signal Analysis for Object Differentiation</li>"
        "<li>Carry out Complex Signal Analysis for Object Detection</li>"
        "<li>Create Machine Learning Models for Object Differentiation &amp; Object Detection using a Large Dataset</li>"
        "<li>Communicate with Red Pitaya Sensor<br>"
        "  &bull; Visualize Real-Time FFT data from Ultrasonic Sensor<br>"
        "  &bull; Log Measurements<br>"
        "  &bull; Predict FFT Class for Object Differentiation<br>"
        "  &bull; Predict Object Distance</li>"
        "</ul>"
    )

    description1.setStyleSheet("font-size: 18px;")
    description1.setWordWrap(True)

    separator = QLabel("<hr>")
    separator.setTextFormat(Qt.RichText)

    description2 = QLabel(
            "<b>Version:</b> v1.0 - 2025 <br>"
            "<b>Developer:</b> Gaurav Honnavara Manjunath<br><br>"
            "For Source Code: <a href='https://github.com/gauravhm96/Ultrasonic-Sensor-ML'>GitHub Repository</a>."
        )

    description2.setStyleSheet("font-size: 18px;")
    description2.setOpenExternalLinks(True)
    description2.setWordWrap(True)

    cwd = os.getcwd()
    logo_path = os.path.join(cwd, "Logo", "UltrasonicApp")
    logo_pixmap = QPixmap(logo_path)
    logo_label = QLabel()
    if not logo_pixmap.isNull():
        logo_label.setPixmap(logo_pixmap.scaled(330, 346, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    else:
        logo_label.setText("Logo not found")
    logo_label.setAlignment(Qt.AlignCenter)

    container = QFrame()
    container.setFrameShape(QFrame.Panel)
    container.setFrameShadow(QFrame.Raised)
    container.setLineWidth(2)

    container_layout = QHBoxLayout()
    container_layout.setSpacing(10)
    container_layout.setContentsMargins(10, 10, 10, 10)
    
    # Create a vertical layout for the descriptions
    text_layout = QVBoxLayout()
    text_layout.setSpacing(10)
    text_layout.addWidget(description1)
    text_layout.addWidget(separator)
    text_layout.addWidget(description2)
    text_layout.addStretch()
    
    # Add text layout on the left and logo on the right
    container_layout.addLayout(text_layout)
    container_layout.addWidget(logo_label)
    container.setLayout(container_layout)
    
    # Finally, add the container to the main layout
    layout.addWidget(container)