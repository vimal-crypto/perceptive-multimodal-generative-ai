import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from .ocr_extractor import extract_text_from_image, save_extracted_text


class OCRWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image_path, preprocess):
        super().__init__()
        self.image_path = image_path
        self.preprocess = preprocess

    def run(self):
        try:
            text = extract_text_from_image(self.image_path, preprocess=self.preprocess)
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class OCRExtractorUI(QWidget):
    """PyQt5 UI for the OCR Text Extraction feature."""

    def __init__(self):
        super().__init__()
        self.image_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PMG-AI: OCR Text Extractor")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("OCR Text Extraction")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #cba6f7;")
        layout.addWidget(title)

        # Image selection
        self.img_label = QLabel("No image selected")
        self.img_label.setStyleSheet("background-color: #313244; padding: 8px; border-radius: 4px;")
        select_btn = QPushButton("Select Image")
        select_btn.setStyleSheet(self._btn("#89b4fa"))
        select_btn.clicked.connect(self.select_image)

        # Preview
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(200)
        self.preview.setStyleSheet("background-color: #313244; border-radius: 6px;")

        # Preprocess toggle
        self.preprocess_chk = QCheckBox("Apply Preprocessing (recommended)")
        self.preprocess_chk.setChecked(True)
        self.preprocess_chk.setStyleSheet("color: #cdd6f4;")

        # Extract button
        extract_btn = QPushButton("Extract Text")
        extract_btn.setFont(QFont("Segoe UI", 13, QFont.Bold))
        extract_btn.setStyleSheet(self._btn("#a6e3a1"))
        extract_btn.clicked.connect(self.extract_text)

        # Result area
        self.result_area = QTextEdit()
        self.result_area.setPlaceholderText("Extracted text will appear here...")
        self.result_area.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 6px; padding: 8px;")
        self.result_area.setMinimumHeight(150)

        # Save button
        save_btn = QPushButton("Save Text")
        save_btn.setStyleSheet(self._btn("#f9e2af"))
        save_btn.clicked.connect(self.save_text)

        for w in [self.img_label, select_btn, self.preview,
                  self.preprocess_chk, extract_btn,
                  self.result_area, save_btn]:
            layout.addWidget(w)

        self.setLayout(layout)

    def _btn(self, color):
        return f"QPushButton {{ background-color: {color}; color: #1e1e2e; font-weight: bold; border-radius: 6px; padding: 8px; }}"

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.image_path = path
            self.img_label.setText(os.path.basename(path))
            pixmap = QPixmap(path)
            self.preview.setPixmap(pixmap.scaled(600, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def extract_text(self):
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return
        self.worker = OCRWorker(self.image_path, self.preprocess_chk.isChecked())
        self.worker.finished.connect(lambda t: self.result_area.setPlainText(t))
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def save_text(self):
        text = self.result_area.toPlainText()
        if not text:
            QMessageBox.warning(self, "Empty", "No text to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Text", "extracted_text.txt", "Text Files (*.txt)")
        if path:
            save_extracted_text(text, path)
            QMessageBox.information(self, "Saved", f"Text saved to {path}")
