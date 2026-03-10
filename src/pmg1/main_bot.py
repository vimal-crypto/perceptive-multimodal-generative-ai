import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from .comic_generation.comic_generator_ui import ComicGeneratorUI
from .style_transfer.style_transfer_ui import StyleTransferUI
from .ocr_text_extraction.ocr_ui import OCRExtractorUI


class MainBotUI(QMainWindow):
    """
    PMG-1 Main application window.
    Provides navigation between: Comic Generator, Style Transfer, OCR Extractor.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Perceptive Multimodal Generative AI — PMG-1")
        self.setMinimumSize(1000, 750)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI';")
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar navigation
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background-color: #181825;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 24, 12, 24)
        sidebar_layout.setSpacing(10)

        logo = QLabel("PMG-AI")
        logo.setFont(QFont("Segoe UI", 18, QFont.Bold))
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("color: #cba6f7; margin-bottom: 20px;")
        sidebar_layout.addWidget(logo)

        self.stacked = QStackedWidget()
        pages = [
            ("Comic Generator", ComicGeneratorUI()),
            ("Style Transfer", StyleTransferUI()),
            ("OCR Extractor", OCRExtractorUI()),
        ]

        for i, (label, widget) in enumerate(pages):
            btn = QPushButton(label)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #313244;
                    color: #cdd6f4;
                    border-radius: 8px;
                    padding: 10px;
                    text-align: left;
                    font-size: 13px;
                }
                QPushButton:hover { background-color: #45475a; }
                QPushButton:checked { background-color: #cba6f7; color: #1e1e2e; font-weight: bold; }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, idx=i: self.switch_page(idx))
            sidebar_layout.addWidget(btn)
            self.stacked.addWidget(widget)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stacked)

    def switch_page(self, index):
        self.stacked.setCurrentIndex(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainBotUI()
    window.show()
    sys.exit(app.exec_())
