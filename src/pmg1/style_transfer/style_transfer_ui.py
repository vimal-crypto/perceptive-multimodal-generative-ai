from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QProgressDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from style_transfer import perform_style_transfer

class StyleTransferWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, content_image, style_image):
        super().__init__()
        self.content_image = content_image
        self.style_image = style_image

    def run(self):
        result_image_path = perform_style_transfer(self.content_image, self.style_image)
        self.finished.emit(result_image_path)

class StyleTransferUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Style Transfer")
        self.setFixedSize(800, 600)
        layout = QVBoxLayout()
        self.content_image_label = QLabel("Content Image: Not selected")
        self.style_image_label = QLabel("Style Image: Not selected")
        self.result_image_label = QLabel("")
        self.content_image_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        self.style_image_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        self.result_image_label.setStyleSheet("font-size: 14px; color: #555; margin: 20px;")
        content_button = QPushButton("Select Content Image")
        style_button = QPushButton("Select Style Image")
        submit_button = QPushButton("Submit")
        return_button = QPushButton("Return")
        button_style = "QPushButton { background-color: #007bff; color: white; font-size: 16px; border-radius: 5px; } QPushButton:hover { background-color: #0056b3; }"
        content_button.setStyleSheet(button_style)
        style_button.setStyleSheet(button_style)
        submit_button.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-size: 16px; border-radius: 5px; } QPushButton:hover { background-color: #218838; }")
        return_button.setStyleSheet(button_style)
        content_button.clicked.connect(self.select_content_image)
        style_button.clicked.connect(self.select_style_image)
        submit_button.clicked.connect(self.start_style_transfer)
        return_button.clicked.connect(self.return_to_upload)
        layout.addWidget(self.content_image_label)
        layout.addWidget(content_button)
        layout.addWidget(self.style_image_label)
        layout.addWidget(style_button)
        layout.addWidget(submit_button)
        layout.addWidget(self.result_image_label)
        layout.addWidget(return_button)
        self.setLayout(layout)

    def select_content_image(self):
        content_image, _ = QFileDialog.getOpenFileName(self, "Select Content Image", "", "Images (*.png *.jpg)")
        if content_image:
            self.content_image_label.setText(f"Content Image: {content_image}")
            self.content_image_path = content_image

    def select_style_image(self):
        style_image, _ = QFileDialog.getOpenFileName(self, "Select Style Image", "", "Images (*.png *.jpg)")
        if style_image:
            self.style_image_label.setText(f"Style Image: {style_image}")
            self.style_image_path = style_image

    def start_style_transfer(self):
        if not hasattr(self, 'content_image_path') or not hasattr(self, 'style_image_path'):
            self.result_image_label.setText("Please select both images.")
            return
        self.progress_dialog = QProgressDialog("Processing...", None, 0, 100, self)
        self.progress_dialog.setWindowTitle("Loading")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        self.worker = StyleTransferWorker(self.content_image_path, self.style_image_path)
        self.worker.finished.connect(self.show_result)
        self.worker.start()

    def show_result(self, result_image_path):
        self.progress_dialog.close()
        pixmap = QPixmap(result_image_path)
        self.result_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def return_to_upload(self):
        self.result_image_label.clear()
        self.content_image_label.setText("Content Image: Not selected")
        self.style_image_label.setText("Style Image: Not selected")

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = StyleTransferUI()
    window.show()
    sys.exit(app.exec_())
