import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import pipeline as hf_pipeline

_device = "cuda" if torch.cuda.is_available() else "cpu"
_sd_pipeline = None
_sentiment_pipeline = None


def _load_sd():
    global _sd_pipeline
    if _sd_pipeline is None:
        print("[INFO] Loading Stable Diffusion for animation...")
        _sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        ).to(_device)
        _sd_pipeline.safety_checker = None
    return _sd_pipeline


def _load_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = hf_pipeline("sentiment-analysis", device=0 if _device == "cuda" else -1)
    return _sentiment_pipeline


def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of a text and return 'positive', 'negative', or 'neutral'.

    Args:
        text: Input text string.

    Returns:
        Sentiment label string.
    """
    sentiment = _load_sentiment()(text)[0]
    label = sentiment['label'].lower()
    if 'positive' in label:
        return 'positive'
    elif 'negative' in label:
        return 'negative'
    return 'neutral'


def sentiment_to_visual_modifier(sentiment: str) -> str:
    """
    Map a sentiment label to a visual style modifier for prompt enhancement.

    Args:
        sentiment: 'positive', 'negative', or 'neutral'.

    Returns:
        Visual modifier string to append to the image prompt.
    """
    modifiers = {
        'positive': 'bright, vibrant colors, warm sunlight, hopeful atmosphere',
        'negative': 'dark, moody, storm clouds, somber tones',
        'neutral': 'natural lighting, balanced composition, realistic'
    }
    return modifiers.get(sentiment, modifiers['neutral'])


def generate_animation(
    prompts: list,
    narratives: list = None,
    output_dir: str = "outputs/animation",
    fps: int = 4,
    use_sentiment: bool = True
) -> str:
    """
    Generate an animated GIF from a sequence of text prompts.
    Optionally modulates visual style using VAD (Valence-Arousal-Dominance)
    sentiment analysis on accompanying narrative text.

    Args:
        prompts: List of scene description strings.
        narratives: Optional list of narrative texts for sentiment analysis.
        output_dir: Directory to save frames and final GIF.
        fps: Frames per second for the output GIF.
        use_sentiment: Whether to use sentiment-based visual modifiers.

    Returns:
        Path to the saved animated GIF.
    """
    pipeline = _load_sd()
    os.makedirs(output_dir, exist_ok=True)
    frames = []

    for i, prompt in enumerate(prompts):
        enhanced_prompt = prompt
        if use_sentiment and narratives and i < len(narratives):
            sentiment = analyze_sentiment(narratives[i])
            modifier = sentiment_to_visual_modifier(sentiment)
            enhanced_prompt = f"{prompt}, {modifier}"
            print(f"[INFO] Frame {i+1}: sentiment={sentiment}, prompt='{enhanced_prompt}'")
        else:
            print(f"[INFO] Frame {i+1}: prompt='{enhanced_prompt}'")

        with torch.autocast(_device) if _device == "cuda" else torch.no_grad():
            frame = pipeline(enhanced_prompt).images[0]

        frame_path = os.path.join(output_dir, f"frame_{i+1:03d}.png")
        frame.save(frame_path)
        frames.append(frame)

    gif_path = os.path.join(output_dir, "animation.gif")
    duration_ms = int(1000 / fps)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms
    )
    print(f"[INFO] Animation saved: {gif_path}")
    return gif_path


try:
    import sys
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QPushButton, QLabel,
        QTextEdit, QLineEdit, QSpinBox, QHBoxLayout,
        QFileDialog, QMessageBox, QApplication
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont

    class AnimationWorker(QThread):
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, prompts, narratives, fps):
            super().__init__()
            self.prompts = prompts
            self.narratives = narratives
            self.fps = fps

        def run(self):
            try:
                path = generate_animation(self.prompts, self.narratives, fps=self.fps)
                self.finished.emit(path)
            except Exception as e:
                self.error.emit(str(e))

    class AnimationGeneratorUI(QWidget):
        """PyQt5 UI for the Animation Generator."""

        def __init__(self):
            super().__init__()
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("PMG-AI: Animation Generator")
            self.setMinimumSize(800, 600)
            self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
            layout = QVBoxLayout()

            title = QLabel("Animation Generator")
            title.setFont(QFont("Segoe UI", 18, QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color: #cba6f7;")
            layout.addWidget(title)

            layout.addWidget(QLabel("Prompts (one per line):"))
            self.prompts_input = QTextEdit()
            self.prompts_input.setPlaceholderText("A sunrise over mountains\nA hero running through the city\n...")
            self.prompts_input.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 4px; padding: 6px;")
            layout.addWidget(self.prompts_input)

            fps_row = QHBoxLayout()
            fps_row.addWidget(QLabel("FPS:"))
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 30)
            self.fps_spin.setValue(4)
            self.fps_spin.setStyleSheet("background-color: #313244; color: #cdd6f4;")
            fps_row.addWidget(self.fps_spin)
            fps_row.addStretch()
            layout.addLayout(fps_row)

            gen_btn = QPushButton("Generate Animation")
            gen_btn.setStyleSheet("QPushButton { background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; border-radius: 6px; padding: 8px; }")
            gen_btn.clicked.connect(self.start_generation)
            layout.addWidget(gen_btn)

            self.status_label = QLabel("")
            self.status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.status_label)

            self.setLayout(layout)

        def start_generation(self):
            prompts = [p.strip() for p in self.prompts_input.toPlainText().splitlines() if p.strip()]
            if not prompts:
                QMessageBox.warning(self, "Error", "Please enter at least one prompt.")
                return
            self.worker = AnimationWorker(prompts, None, self.fps_spin.value())
            self.worker.finished.connect(lambda p: self.status_label.setText(f"Saved: {p}"))
            self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
            self.worker.start()
            self.status_label.setText("Generating...")

except ImportError:
    class AnimationGeneratorUI:
        """Placeholder when PyQt5 is not available."""
        pass
