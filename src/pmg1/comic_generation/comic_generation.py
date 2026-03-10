import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QLineEdit, QFileDialog, QProgressDialog, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from comic_generator_logic import generate_comic

class ComicWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, prompt, num_panels):
        super().__init__()
        self.prompt = prompt
        self.num_panels = num_panels

    def run(self):
        output_path = generate_comic(self.prompt, self.num_panels)
        self.finished.emit(output_path)

class ComicGenerationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Comic Generation")
        self.setFixedSize(800, 600)
        layout = QVBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your comic story prompt...")
        self.prompt_input.setStyleSheet("font-size:14px; padding:8px;")
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        generate_btn = QPushButton("Generate Comic")
        generate_btn.setStyleSheet("font-size:16px; padding:10px; background:#007bff; color:white; border-radius:5px;")
        generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(QLabel("Story Prompt:"))
        layout.addWidget(self.prompt_input)
        layout.addWidget(generate_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def start_generation(self):
        prompt = self.prompt_input.text()
        if not prompt:
            self.result_label.setText("Please enter a prompt.")
            return
        self.progress = QProgressDialog("Generating comic...", None, 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()
        self.worker = ComicWorker(prompt, num_panels=4)
        self.worker.finished.connect(self.show_result)
        self.worker.start()

    def show_result(self, output_path):
        self.progress.close()
        pixmap = QPixmap(output_path)
        self.result_label.setPixmap(pixmap.scaled(700, 400, Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ComicGenerationUI()
    window.show()
    sys.exit(app.exec_())
