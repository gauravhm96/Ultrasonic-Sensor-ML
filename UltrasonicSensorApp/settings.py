from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QLabel,QPushButton, QFileDialog,QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from file_system import FileHandler


def Left_section():
    left_section = QWidget()
    #left_section.setStyleSheet("border: 2px solid black; background-color: lightgray;")
    
    # Create a vertical layout for the left section
    left_layout = QVBoxLayout()
    left_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    # Add a logo at the top
    logo_label = QLabel()
    logo_path = r"E:\Frankfurt University of Applied Sciences\Master Thesis\Code\Example\Logo\image(1).png"
    logo_pixmap = QPixmap(logo_path)
    logo_label.setPixmap(logo_pixmap)
    logo_label.setScaledContents(True)
    logo_label.setFixedSize(300, 120)
    logo_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    # Add spacer to create space between the logo and the button
    spacer = QSpacerItem(100, 50, QSizePolicy.Minimum, QSizePolicy.Fixed) 

    # Add a "Select File" button
    select_file_button = QPushButton("Select File")
    select_file_button.setStyleSheet("background-color: white; border: 1px solid black; padding: 5px;")
    select_file_button.setFixedWidth(150)

    # Add a label to display the selected file path
    file_path_label = QLabel("No file selected")
    file_path_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 10px; color: black; font-size: 12px;")
    file_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Allow it to expand horizontally
    file_path_label.setWordWrap(True)  # Allow multiline for long paths
     
    # Create an instance of FileHandler
    file_handler = FileHandler()

    # Connect the button click to the select_file method
    select_file_button.clicked.connect(lambda: update_file_path(file_handler, file_path_label))

    # Add the logo to the layout
    left_layout.addWidget(logo_label)
    left_layout.addSpacerItem(spacer)  # Add space below the logo
    left_layout.addWidget(select_file_button)
    left_layout.addWidget(file_path_label)

    # Set the layout to the left section
    left_section.setLayout(left_layout)

    return left_section


def Right_section():
    """Creates and returns the right section with fixed style."""
    right_section = QWidget()
    right_section.setStyleSheet("border: 2px solid black; background-color: lightblue;")
    
    return right_section

def setup_gui(main_window):
    main_window.setWindowTitle("Ultrasonic Sensor App")
    main_window.setMinimumSize(1200, 800)
    main_window.setGeometry(100, 100, 1000, 200)  # Set initial window size

    # Main horizontal layout to hold left and right sections
    main_layout = QHBoxLayout()

    # Create left and right sections
    left_section = Left_section()
    right_section = Right_section()

    # Add sections to the main layout
    main_layout.addWidget(left_section, 30)  # 40% of the space
    main_layout.addWidget(right_section, 70)  # 60% of the space

    # Set the layout to the main window
    container = QWidget()
    container.setLayout(main_layout)
    main_window.setCentralWidget(container)

    return main_window

def update_file_path(file_handler, file_path_label):
    """Handle the file selection and update the file path label."""
    selected_file = file_handler.select_file()
    if selected_file:
        file_path_label.setText(selected_file)
    else:
        file_path_label.setText("No file selected")
