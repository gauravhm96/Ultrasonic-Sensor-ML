from PyQt5.QtWidgets import QFileDialog


class FileHandler:
    def __init__(self):
        self.file_path = None

    def select_file(self, parent=None):
        """Open a file dialog to select a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Select Signal File", "", "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if file_path:
            self.file_path = file_path
            return file_path
        return None
