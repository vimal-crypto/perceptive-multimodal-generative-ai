import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressDialog, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
from .style_transfer import perform_style_transfer


class StyleTransferWorker(QThread):
    """Background thread that runs neural style transfer without blocking the UI."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, content_image: str, style_image: str, epochs: int, steps: int):
        super().__init__()
        self.content_image = content_image
        self.style_image = style_image
        self.epochs = epochs
        self.steps = steps

    def run(self):
        try:
            result_path = perform_style_transfer(
                self.content_image,
                self.style_image,
                output_path="outputs/enhanced-stylized-image.png",
                epochs=self.epochs,
                steps_per_epoch=self.steps
            )
            self.finished.emit(result_path)
        except Exception as e:
            self.error.emit(str(e))


class StyleTransferUI(QWidget):
    """PyQt5 UI for the Image Style Transfer feature of PMG-AI."""

    def __init__(self):
        super().__init__()
        self.content_image_path = None
        self.style_image_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PMG-AI: Image Style Transfer")
        self.setMinimumSize(900, 650)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI';")

        layout = QVBoxLayout()
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Image Style Transfer")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #cba6f7; margin-bottom: 10px;")
        layout.addWidget(title)

        # Image selection row
        img_row = QHBoxLayout()
        self.content_label = QLabel("Content Image: Not selected")
        self.style_label = QLabel("Style Image: Not selected")
        for lbl in [self.content_label, self.style_label]:
            lbl.setStyleSheet("background-color: #313244; padding: 8px; border-radius: 4px;")

        content_btn = QPushButton("Select Content Image")
        style_btn = QPushButton("Select Style Image")
        content_btn.setStyleSheet(self._btn_style("#89b4fa"))
        style_btn.setStyleSheet(self._btn_style("#89b4fa"))
        content_btn.clicked.connect(self.select_content_image)
        style_btn.clicked.connect(self.select_style_image)

        left_col = QVBoxLayout()
        left_col.addWidget(self.content_label)
        left_col.addWidget(content_btn)

        right_col = QVBoxLayout()
        right_col.addWidget(self.style_label)
        right_col.addWidget(style_btn)

        img_row.addLayout(left_col)
        img_row.addLayout(right_col)
        layout.addLayout(img_row)

        # Epochs slider
        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Epochs:"))
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setRange(1, 10)
        self.epoch_slider.setValue(3)
        self.epoch_slider.setStyleSheet("color: #cba6f7;")
        self.epoch_label = QLabel("3")
        self.epoch_slider.valueChanged.connect(lambda v: self.epoch_label.setText(str(v)))
        epoch_row.addWidget(self.epoch_slider)
        epoch_row.addWidget(self.epoch_label)
        layout.addLayout(epoch_row)

        # Submit button
        submit_btn = QPushButton("Run Style Transfer")
        submit_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        submit_btn.setFixedHeight(45)
        submit_btn.setStyleSheet(self._btn_style("#a6e3a1"))
        submit_btn.clicked.connect(self.start_style_transfer)
        layout.addWidget(submit_btn)

        # Result preview
        self.result_label = QLabel("Stylized image will appear here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(280)
        self.result_label.setStyleSheet("background-color: #313244; border-radius: 8px; color: #6c7086;")
        layout.addWidget(self.result_label)

        # Return button
        return_btn = QPushButton("Clear / Return")
        return_btn.setStyleSheet(self._btn_style("#f38ba8"))
        return_btn.clicked.connect(self.clear_ui)
        layout.addWidget(return_btn)

        self.setLayout(layout)

    def _btn_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: #1e1e2e;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
            }}
        """

    def select_content_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Content Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.content_image_path = path
            self.content_label.setText(f"Content: {os.path.basename(path)}")

    def select_style_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Style Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.style_image_path = path
            self.style_label.setText(f"Style: {os.path.basename(path)}")

    def start_style_transfer(self):
        if not self.content_image_path or not self.style_image_path:
            QMessageBox.warning(self, "Error", "Please select both content and style images.")
            return

        self.progress = QProgressDialog("Running style transfer...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()

        self.worker = StyleTransferWorker(
            self.content_image_path,
            self.style_image_path,
            epochs=self.epoch_slider.value(),
            steps=50
        )
        self.worker.finished.connect(self.show_result)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def show_result(self, result_path):
        self.progress.close()
        pixmap = QPixmap(result_path)
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.width(), self.result_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def on_error(self, msg):
        self.progress.close()
        QMessageBox.critical(self, "Error", f"Style transfer failed: {msg}")

    def clear_ui(self):
        self.result_label.clear()
        self.result_label.setText("Stylized image will appear here")
        self.content_label.setText("Content Image: Not selected")
        self.style_label.setText("Style Image: Not selected")
        self.content_image_path = None
        self.style_image_path = None


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = StyleTransferUI()
    window.show()
    sys.exit(app.exec_())
