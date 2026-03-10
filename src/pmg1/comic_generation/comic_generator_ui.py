import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QProgressDialog, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
from .comic_generator_logic import generate_comic_panel, create_comic_strip


class ComicWorker(QThread):
    """Background thread for comic generation to keep the UI responsive."""
    finished = pyqtSignal(list)  # emits list of generated image paths
    error = pyqtSignal(str)

    def __init__(self, prompts, dialogues):
        super().__init__()
        self.prompts = prompts
        self.dialogues = dialogues

    def run(self):
        try:
            panel_paths = []
            for i, (prompt, dialogue) in enumerate(zip(self.prompts, self.dialogues)):
                output_path = f"outputs/comic_panel_{i+1}.png"
                os.makedirs("outputs", exist_ok=True)
                generate_comic_panel(prompt, dialogue, output_path)
                panel_paths.append(output_path)
            self.finished.emit(panel_paths)
        except Exception as e:
            self.error.emit(str(e))


class ComicGeneratorUI(QWidget):
    """Main UI widget for the Comic Generation module."""

    def __init__(self):
        super().__init__()
        self.panel_inputs = []  # list of (prompt_input, dialogue_input) tuples
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PMG-AI: Comic Generator")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI';")

        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Comic Generator")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #cba6f7; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Number of panels
        panel_row = QHBoxLayout()
        panel_label = QLabel("Number of Panels:")
        self.panel_count = QSpinBox()
        self.panel_count.setRange(1, 10)
        self.panel_count.setValue(3)
        self.panel_count.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 4px;")
        gen_panels_btn = QPushButton("Generate Panel Inputs")
        gen_panels_btn.setStyleSheet(self._btn_style("#89b4fa"))
        gen_panels_btn.clicked.connect(self.generate_panel_inputs)
        panel_row.addWidget(panel_label)
        panel_row.addWidget(self.panel_count)
        panel_row.addWidget(gen_panels_btn)
        panel_row.addStretch()
        main_layout.addLayout(panel_row)

        # Dynamic panel input area
        self.panels_container = QVBoxLayout()
        main_layout.addLayout(self.panels_container)

        # Generate button
        self.generate_btn = QPushButton("Generate Comic")
        self.generate_btn.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.generate_btn.setFixedHeight(45)
        self.generate_btn.setStyleSheet(self._btn_style("#a6e3a1"))
        self.generate_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.generate_btn)

        # Preview label
        self.preview_label = QLabel("Comic preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("background-color: #313244; border-radius: 8px; color: #6c7086;")
        main_layout.addWidget(self.preview_label)

        # Save button
        self.save_btn = QPushButton("Save Comic Strip")
        self.save_btn.setStyleSheet(self._btn_style("#f38ba8"))
        self.save_btn.clicked.connect(self.save_comic)
        self.save_btn.setEnabled(False)
        main_layout.addWidget(self.save_btn)

        self.setLayout(main_layout)
        self.generated_panels = []

        # Initialize with default panels
        self.generate_panel_inputs()

    def _btn_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: #1e1e2e;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background-color: #45475a; color: #6c7086; }}
        """

    def generate_panel_inputs(self):
        # Clear existing
        while self.panels_container.count():
            item = self.panels_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.panel_inputs.clear()

        n = self.panel_count.value()
        for i in range(n):
            row = QHBoxLayout()
            prompt_input = QLineEdit()
            prompt_input.setPlaceholderText(f"Panel {i+1} scene prompt (e.g. 'A hero flying over a city')")
            prompt_input.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 6px; border-radius: 4px;")

            dialogue_input = QLineEdit()
            dialogue_input.setPlaceholderText(f"Panel {i+1} dialogue (or leave blank to auto-generate)")
            dialogue_input.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 6px; border-radius: 4px;")

            row.addWidget(QLabel(f"Panel {i+1}:"))
            row.addWidget(prompt_input)
            row.addWidget(dialogue_input)

            container = QWidget()
            container.setLayout(row)
            self.panels_container.addWidget(container)
            self.panel_inputs.append((prompt_input, dialogue_input))

    def start_generation(self):
        prompts = [p.text().strip() for p, _ in self.panel_inputs]
        dialogues = [d.text().strip() for _, d in self.panel_inputs]

        if not all(prompts):
            QMessageBox.warning(self, "Input Error", "Please fill in all panel prompts.")
            return

        self.generate_btn.setEnabled(False)
        self.progress = QProgressDialog("Generating comic panels...", None, 0, 0, self)
        self.progress.setWindowTitle("Please Wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()

        self.worker = ComicWorker(prompts, dialogues)
        self.worker.finished.connect(self.on_generation_complete)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_complete(self, panel_paths):
        self.progress.close()
        self.generate_btn.setEnabled(True)
        self.generated_panels = panel_paths

        strip_path = "outputs/comic_strip.png"
        create_comic_strip(panel_paths, strip_path)

        pixmap = QPixmap(strip_path)
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.save_btn.setEnabled(True)

    def on_generation_error(self, error_msg):
        self.progress.close()
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Generation failed: {error_msg}")

    def save_comic(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Comic Strip", "comic_strip.png", "Images (*.png *.jpg)")
        if save_path and self.generated_panels:
            create_comic_strip(self.generated_panels, save_path)
            QMessageBox.information(self, "Saved", f"Comic saved to {save_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ComicGeneratorUI()
    window.show()
    sys.exit(app.exec_())
