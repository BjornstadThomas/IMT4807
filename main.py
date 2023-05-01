import sys
import os
import time
import numpy as np
import torch
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QLineEdit, QCheckBox, QProgressBar,
                             QPushButton, QFileDialog, QGraphicsScene, QGraphicsView, QWidget, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# Import stylegan2-ada-pytorch dependencies
sys.path.append("stylegan2")
import stylegan2.dnnlib as dnnlib
import stylegan2.legacy as legacy

# Load pre-trained StyleGAN2 model
network_pkl = 'C:\\Users\\Thomas\\PycharmProjects\\scientificProject\\models\\network-snapshot-000200.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with dnnlib.util.open_url(network_pkl) as fp:
    G = None #legacy.load_network_pkl(fp)['G_ema'].to(device)


class CustomSlider(QHBoxLayout):
    def __init__(self, label_text, min_val, max_val, default_val, decimals=0):
        super().__init__()
        self.label = QLabel(label_text)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default_val)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval((max_val - min_val) // 10)

        self.value_input = QSpinBox()
        self.value_input.setMinimum(min_val)
        self.value_input.setMaximum(max_val)
        self.value_input.setValue(default_val)
        self.slider.valueChanged.connect(self.update_value)
        self.value_input.valueChanged.connect(self.update_value)

        self.addWidget(self.label)
        self.addWidget(self.slider)
        self.addWidget(self.value_input)
        self.decimals = decimals

    def update_value(self, value):
        self.slider.setValue(value)
        self.value_input.setValue(value)

    def set_visible(self, visible):
        self.label.setVisible(visible)
        self.slider.setVisible(visible)
        self.value_input.setVisible(visible)

class ImaGenie(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('ImaGenie - Image Generation Tool')
        self.setGeometry(100, 100, 1200, 800)

        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()

        self.browse_button = QPushButton("Browse Model", self)
        self.browse_button.clicked.connect(self.browse_model)
        vbox.addWidget(self.browse_button)

        self.model_path_label = QLabel("Model Path: None")
        vbox.addWidget(self.model_path_label)

        hbox = QHBoxLayout()
        self.num_images_slider = CustomSlider("Number of Images (1-1000):", 1, 1000, 1)
        hbox.addLayout(self.num_images_slider)
        self.num_images_slider.slider.valueChanged.connect(self.update_seeds_input_state)

        self.seeds_input = QLineEdit(self)
        self.seeds_input.setPlaceholderText("Enter specific seeds, e.g. 1,2,3-5")
        hbox.addWidget(self.seeds_input)
        vbox.addLayout(hbox)
        self.seeds_input.setVisible(False)

        self.truncation_slider = CustomSlider("Truncation (1-100):", 1, 100, 50)
        vbox.addLayout(self.truncation_slider)
        self.truncation_slider.set_visible(False)


        self.variables_checkbox = QCheckBox("Options", self)
        self.variables_checkbox.stateChanged.connect(self.toggle_variables_state)
        self.variables_checkbox.stateChanged.connect(self.toggle_options_visibility)
        vbox.addWidget(self.variables_checkbox)

        self.variables_group = []

        # Add more sliders and text fields for generator variables here
        # Example: custom_slider = CustomSlider("Variable Name (min-max):", min_val, max_val, default_val, decimals=2)
        # self.variables_group.append(custom_slider)
        # vbox.addWidget(custom_slider)

        for variable in self.variables_group:
            variable.setEnabled(False)

        gen_button = QPushButton('Generate Images', self)
        gen_button.clicked.connect(self.generate_images)
        vbox.addWidget(gen_button)

        self.image_display = QLabel(self)
        vbox.addWidget(self.image_display)

        hbox = QHBoxLayout()
        prev_button = QPushButton('Previous Image', self)
        prev_button.clicked.connect(self.previous_image)
        hbox.addWidget(prev_button)
        next_button = QPushButton('Next Image', self)
        next_button.clicked.connect(self.next_image)
        hbox.addWidget(next_button)
        vbox.addLayout(hbox)

        #PROGRESS BAR#
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #FFFFFF;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.progress_bar.setMinimumHeight(20)
        vbox.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)

        self.images = []
        self.current_image_index = 0
        self.seed = 0

    def browse_model(self):
        global G  # Add this line to access the global G variable

        model_path, _ = QFileDialog.getOpenFileName(self, 'Open Model', '', 'Pickle Files (*.pkl);;All Files (*)')
        if model_path:
            with open(model_path, 'rb') as fp:
                G = legacy.load_network_pkl(fp)['G_ema'].to(device)  # Update this line
            self.model_path_label.setText(f"Model Path: {model_path}")

    def update_seeds_input_state(self):
        if self.num_images_slider.slider.value() > 1:
            self.seeds_input.setDisabled(True)
        else:
            self.seeds_input.setDisabled(False)

    def toggle_variables_state(self, state):
        for variable in self.variables_group:
            variable.setEnabled(state)

    def generate_images(self):
        global G
        if G is None:
            self.show_error_message("No Model Loaded", "Please load a model before generating images.")
            return

        if self.num_images_slider.value_input.value() > 1:
            seeds = range(self.seed, self.seed + self.num_images_slider.value_input.value())
        else:
            seeds = [int(seed) for seed in self.seeds_input.text().split(',') if '-' not in seed] + \
                    [int(seed.strip()) for seed_range in self.seeds_input.text().split(',') if '-' in seed_range
                     for seed in range(int(seed_range.split('-')[0]), int(seed_range.split('-')[1]) + 1)]

        # Rest of the function remains the same

        self.progress_bar.setMaximum(len(seeds))
        self.progress_bar.setValue(0)

        for i, seed in enumerate(seeds):
            # Generate images
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, None, truncation_psi=self.truncation_slider.slider.value() / 100)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            pil_img = Image.fromarray(img[0].cpu().numpy(), 'RGB')

            self.images.append(pil_img)

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.seed += self.num_images_slider.slider.value()

        # Save generated images
        model_name = os.path.splitext(os.path.basename(self.model_path_label.text().replace("Model Path: ", "")))[0]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        #timestamp = time.strftime("%Y.%m.%d-%H:%M:%S")
        output_dir = os.path.join("result", f"{model_name}_{timestamp}")
        self.save_images(self.images, output_dir)

        self.display_images()

    def display_images(self):
        if self.images:
            pixmap = QPixmap.fromImage(ImageQt.ImageQt(self.images[self.current_image_index]))
            pixmap = pixmap.scaled(self.image_display.width(), self.image_display.height(), Qt.KeepAspectRatio)
            self.image_display.setPixmap(pixmap)
        else:
            self.image_display.clear()

    def previous_image(self):
        if self.images:
            self.current_image_index -= 1
            if self.current_image_index < 0:
                self.current_image_index = len(self.images) - 1
            self.display_images()

    def next_image(self):
        if self.images:
            self.current_image_index += 1
            if self.current_image_index >= len(self.images):
                self.current_image_index = 0
            self.display_images()

    def show_error_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

    def save_images(self, images, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f'image_{i + 1}.png')
            image.save(image_path)

    def toggle_options_visibility(self, state):
        self.truncation_slider.set_visible(state)
        self.seeds_input.setVisible(state)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    main = ImaGenie()
    main.show()
    sys.exit(app.exec_())


