import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from settings import setup_gui
from file_system import FileHandler


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the GUI layout
        self.ui = setup_gui(self)

    def run(self):
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.run()
    sys.exit(app.exec_())
