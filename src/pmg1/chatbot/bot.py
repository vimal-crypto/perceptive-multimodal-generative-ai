import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from style_transfer_ui import StyleTransferUI
from comic_generation import ComicGenerationUI

class MainBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perceptive Multimodal Generative AI")
        self.setFixedSize(800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("Perceptive Multimodal Generative AI")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)

        btn_style = QPushButton("Image Style Transfer")
        btn_comic = QPushButton("Comic Generation")

        btn_style.setStyleSheet("font-size:16px; padding:10px; background:#007bff; color:white; border-radius:5px;")
        btn_comic.setStyleSheet("font-size:16px; padding:10px; background:#28a745; color:white; border-radius:5px;")

        btn_style.clicked.connect(self.open_style_transfer)
        btn_comic.clicked.connect(self.open_comic_generation)

        layout.addWidget(btn_style)
        layout.addWidget(btn_comic)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_style_transfer(self):
        self.style_win = StyleTransferUI()
        self.style_win.show()

    def open_comic_generation(self):
        self.comic_win = ComicGenerationUI()
        self.comic_win.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainBot()
    window.show()
    sys.exit(app.exec_())
