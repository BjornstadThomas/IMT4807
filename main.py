import sys
import os
import time
import numpy as np
import torch
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QLineEdit, QCheckBox, QProgressBar,
                             QPushButton, QFileDialog, QGraphicsScene, QGraphicsView,
                             QWidget, QMessageBox, QSpinBox, QRadioButton, QButtonGroup,
                             QToolBar, QAction, QComboBox)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
import subprocess
import configparser
import requests
import tempfile
import traceback


# Import stylegan2-ada-pytorch dependencies
sys.path.append("stylegan2")
import stylegan2.dnnlib as dnnlib
import stylegan2.dnnlib.util
import stylegan2.legacy as legacy

# Load pre-trained StyleGAN2 model
network_pkl = 'C:\\Users\\Thomas\\PycharmProjects\\scientificProject\\models\\network-snapshot-000200.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with dnnlib.util.open_url(network_pkl) as fp:
    G = None #legacy.load_network_pkl(fp)['G_ema'].to(device)


# Read config file
config = configparser.ConfigParser()
config.read('config.ini')
download_models_url = config.get('links', 'download_models_url')

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
        self.setMinimumWidth(400)

        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()

        self.browse_button = QPushButton("Browse Model", self)
        self.browse_button.clicked.connect(self.browse_model)
        vbox.addWidget(self.browse_button)

        self.model_path_label = QLabel("Model Path: None")
        self.model_path_label.setMaximumHeight(50)  # Adjust the value to set the desired height
        font = self.model_path_label.font()
        font.setPointSize(14)  # Set the desired font size
        self.model_path_label.setFont(font)
        vbox.addWidget(self.model_path_label)

        hbox = QHBoxLayout()

        # New code: Download Model Button
        self.download_model_button = QPushButton("Download Model", self)
        self.download_model_button.clicked.connect(self.open_download_page)
        hbox.addWidget(self.download_model_button)

        # New code: Model Dropdown Menu
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.addItem("None")
        self.model_dropdown.addItem("MetFaces")
        self.model_dropdown.addItem("CIFAR-10")
        self.model_dropdown.addItem("FFHQ")
        self.model_dropdown.currentIndexChanged.connect(self.on_model_selected)
        hbox.addWidget(self.model_dropdown)

        hbox2 = QHBoxLayout()

        self.num_images_slider = CustomSlider("Number of Images (1-1000):", 1, 1000, 1)
        hbox2.addLayout(self.num_images_slider)
        self.num_images_slider.slider.valueChanged.connect(self.update_seeds_input_state)

        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)

        self.seeds_input = QLineEdit(self)
        self.seeds_input.setPlaceholderText("Enter specific seeds, e.g. 1,2,3-5")
        self.seeds_input.setMaximumWidth(self.width() * 0.5)
        hbox.addWidget(self.seeds_input)
        vbox.addLayout(hbox)
        self.seeds_input.setVisible(False)

        self.truncation_slider = CustomSlider("Truncation (1-100):", 1, 100, 50)
        vbox.addLayout(self.truncation_slider)
        self.truncation_slider.set_visible(False)

        hbox = QHBoxLayout()
        hbox.setAlignment(Qt.AlignLeft)

        # Add this code snippet to create the noise_mode_label object
        self.noise_mode_label = QLabel("Noise mode:")
        hbox.addWidget(self.noise_mode_label)
        self.noise_mode_label.setVisible(False)  # Initially hide the label

        self.const_noise_mode = QRadioButton("Const")
        self.random_noise_mode = QRadioButton("Random")
        self.none_noise_mode = QRadioButton("None")
        self.noise_mode_group = QButtonGroup(self)
        self.noise_mode_group.addButton(self.const_noise_mode)
        self.noise_mode_group.addButton(self.random_noise_mode)
        self.noise_mode_group.addButton(self.none_noise_mode)
        self.const_noise_mode.setChecked(True)  # Set 'const' as the default option
        hbox.addWidget(self.const_noise_mode)
        hbox.addWidget(self.random_noise_mode)
        hbox.addWidget(self.none_noise_mode)
        vbox.addLayout(hbox)

        self.noise_mode_group.setExclusive(True)
        #self.noise_mode_group.setVisible(True)
        self.const_noise_mode.setVisible(False)
        self.random_noise_mode.setVisible(False)
        self.none_noise_mode.setVisible(False)


        self.variables_checkbox = QCheckBox("Options", self)
        self.variables_checkbox.stateChanged.connect(self.toggle_variables_state)
        self.variables_checkbox.stateChanged.connect(self.toggle_options_visibility)
        vbox.addWidget(self.variables_checkbox)

        self.variables_group = []

        for variable in self.variables_group:
            variable.setEnabled(False)

        gen_button = QPushButton('Generate Images', self)
        gen_button.clicked.connect(self.generate_images)
        vbox.addWidget(gen_button)

        self.image_display = QLabel(self)
        self.image_display.setAlignment(Qt.AlignCenter)
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

        # Add this block of code to create a help button and connect it to the open_help_pdf method
        help_button_action = QAction(QIcon("resources/Help.png"), "Help", self)
        help_button_action.triggered.connect(self.open_user_guide)
        toolbar = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        toolbar.addAction(help_button_action)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

    def browse_model(self):
        global G  # Add this line to access the global G variable

        if self.model_dropdown.currentText() != "None":
            return

        model_path, _ = QFileDialog.getOpenFileName(self, 'Open Model', '', 'Pickle Files (*.pkl);;All Files (*)')
        if model_path:
            with open(model_path, 'rb') as fp:
                G = legacy.load_network_pkl(fp)['G_ema'].to(device)  # Update this line
            self.model_path_label.setText(f"Model Path: {model_path}")

    def open_download_page(self):
        # Update the download_models_url with your desired URL
        download_models_url = "https://example.com/download_models"
        QDesktopServices.openUrl(QUrl(download_models_url))

    def on_model_selected(self, index):
        try:
            if self.model_dropdown.currentText() != "None":
                self.browse_button.setEnabled(False)
            else:
                self.browse_button.setEnabled(True)

            model_name = self.model_dropdown.currentText()
            if index == 0:
                self.model_path_label.setText("Model Path: None")
            else:
                self.model_path_label.setText(f"Selected Model: {model_name}")
                model_url = config.get('models', model_name.lower())
                self.download_and_load_model(model_url)
        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()

    def download_and_load_model(self, model_url):
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "stylegan2")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = dnnlib.util.open_url(model_url, cache_dir=cache_dir, return_filename=True)

        with open(cache_file, "rb") as f:
            self.G = torch.jit.load(f).cuda()

        self.update_generate_images_button()

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
        selected_model = self.model_dropdown.currentText()
        if G is None and selected_model == "None":
            self.show_error_message("No Model Loaded", "Please load a model before generating images.")
            return

        if selected_model != "None":
            # Read the download URL for the selected model from the config.ini file
            model_url = config.get('links', selected_model.lower().replace(" ", "_") + '_url')
            if not model_url:
                self.show_error_message("Invalid Model URL",
                                        "The URL for the selected model is not found in the config.ini file.")
                return

            # Download the model and save it to a temporary file
            model_response = requests.get(model_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
                fp.write(model_response.content)
                model_path = fp.name

            # Load the model
            with open(model_path, 'rb') as fp:
                G = legacy.load_network_pkl(fp)['G_ema'].to(device)
            os.unlink(model_path)  # Delete the temporary model file

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
            #img = G(z, None, truncation_psi=self.truncation_slider.slider.value() / 100, noise_mode=self.noise_mode)
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
        visible = state == Qt.Checked
        self.truncation_slider.set_visible(visible)
        self.seeds_input.setVisible(visible)
        self.const_noise_mode.setVisible(visible)
        self.random_noise_mode.setVisible(visible)
        self.none_noise_mode.setVisible(visible)

        # Add this line to show/hide the "Noise Mode" label
        self.noise_mode_label.setVisible(visible)

    def set_noise_mode(self):
        if self.const_noise_mode.isChecked():
            self.noise_mode = 'const'
        elif self.random_noise_mode.isChecked():
            self.noise_mode = 'random'
        elif self.none_noise_mode.isChecked():
            self.noise_mode = 'none'

    def open_user_guide(self):
        user_guide_path = os.path.join("resources", "user_guide.pdf")
        if not os.path.isfile(user_guide_path):
            self.show_error_message("User Guide Missing",
                                    "The user guide is missing. Please make sure it is located in the 'resources' folder.")
            return

        try:
            if sys.platform == 'win32':
                os.startfile(user_guide_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', user_guide_path])
            else:
                subprocess.Popen(['xdg-open', user_guide_path])
        except Exception as e:
            self.show_error_message("Error Opening User Guide",
                                    f"An error occurred while trying to open the user guide: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon("resources/Logo ImaGenie.jpg"))
    main = ImaGenie()
    main.show()
    sys.exit(app.exec_())


