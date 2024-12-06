import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QTextEdit, QGridLayout
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtCore import Qt

import tensorflow as tf
from inference_sdk import InferenceHTTPClient
from transformers import pipeline

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="8ryrnLvm9tGdLMyEMBlH"
)

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Recipe Recommendations System for Smart Fridge')
        self.setGeometry(100, 100, 1200, 900)  # Larger window size

        # Load a modern font
        try:
            font_id = QFontDatabase.addApplicationFont(":/fonts/OpenSans-Regular.ttf")
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            modern_font = QFont(font_family, 16)
        except IndexError:
            modern_font = QFont("Arial", 16)  # Use a fallback font if loading fails

        central_widget = QWidget()
        layout = QHBoxLayout()  # Horizontal layout

        # Add other widgets to the layout
        layout_right = QVBoxLayout()

        # Add a heading label with modern font
        heading_label = QLabel("Recipe Recommendations System for Smart Fridge")
        heading_label.setFont(QFont(modern_font))
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setStyleSheet("font-size: 24px; color: #333333; margin-bottom: 20px;")  # Styling enhancements
        layout_right.addWidget(heading_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #333333; margin-bottom: 20px;")  # Enhance border and spacing

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold; color: #333333; margin-bottom: 20px;")  # Enhance styling

        self.recipe_text = QTextEdit()
        self.recipe_text.setReadOnly(True)
        self.recipe_text.setStyleSheet("background-color: #F0F0F0; border: 2px solid #333333; padding: 10px;")  # Adjust text area styling

        open_button = QPushButton('Upload Image')
        open_button.setStyleSheet("background-color: #4CAF50; color: #FFFFFF; padding: 10px 20px; border-radius: 8px; margin-bottom: 20px;")  # Enhance button styling
        open_button.clicked.connect(self.open_image)

        layout_right.addWidget(self.result_label, alignment=Qt.AlignCenter)  # Align center
        layout_right.addWidget(QLabel("Recipe Suggestions:"), alignment=Qt.AlignCenter)  # Align center
        layout_right.addWidget(self.recipe_text)
        layout_right.addWidget(open_button, alignment=Qt.AlignCenter)  # Align center

        layout.addLayout(layout_right)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec():
            image_path = file_dialog.selectedFiles()[0]
            self.show_image(image_path)
            self.detect_objects(image_path)

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def detect_objects(self, image_path):
        result = CLIENT.infer(image_path, model_id="aicook-lcv4d/3")
        detected_objects = [prediction['class'] for prediction in result['predictions']]
        self.result_label.setText(', '.join(detected_objects))
        self.get_recipe_suggestions(detected_objects)

    def get_recipe_suggestions(self, ingredients):
        prompt = f"Based on the following ingredients: {', '.join(ingredients)}, please suggest a good recipe."

        # Initialize the text generation pipeline
        generator = pipeline('text-generation', model='gpt2')

        # Generate recipe suggestions
        recipes = generator(prompt, max_length=1024, num_return_sequences=3)

        # Display the recipe suggestions
        self.recipe_text.setPlainText('\n\n'.join([recipe['generated_text'] for recipe in recipes]))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
