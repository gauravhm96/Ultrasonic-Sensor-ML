import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QWidget,
)
from PyQt5.QtCore import Qt


class FileLoaderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Data Loader")
        self.setGeometry(100, 100, 500, 300)

        # Layout
        layout = QVBoxLayout()

        # Label for Instructions
        self.instruction_label = QLabel("Click the button below to select a signal file.")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction_label)

        # Select File Button
        self.select_button = QPushButton("Select File")
        self.select_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.select_button)

        # File Path Label
        self.file_path_label = QLabel("No file selected.")
        self.file_path_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.file_path_label)

        # Data Preview Label
        self.data_preview_label = QLabel("File content will appear here.")
        self.data_preview_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.data_preview_label)

        # Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Signal File", "", "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if file_path:
            self.file_path_label.setText(f"Selected File: {file_path}")
            self.load_and_display_data(file_path)

    def load_and_display_data(self, file_path):
        try:
            # Load data into Pandas DataFrame
            signal_data = pd.read_csv(file_path, delimiter="\t", header=None)

            # Display Data (showing first 5 rows as a preview)
            preview = signal_data.head().to_string(index=False)
            self.data_preview_label.setText(f"Data Preview:\n{preview}")
        except Exception as e:
            self.data_preview_label.setText(f"Error loading file: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileLoaderApp()
    window.show()
    sys.exit(app.exec_())
