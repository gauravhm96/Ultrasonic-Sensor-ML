from PyQt5.QtWidgets import QApplication
from ultrasonic_app import SensorGUI
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Initialize and show the main settings window
    settings_window = SensorGUI()
    settings_window.show()

    # Execute the application
    sys.exit(app.exec_())
