# ✍️ Handwritten Digit Recognizer: End-to-End Deep Learning Web App

A complete, end-to-end Machine Learning project that trains a Convolutional Neural Network (CNN) on the MNIST dataset and deploys it via a Flask REST API connected to an interactive, touch-friendly web-based drawing board.

## ✨ Features
* **Custom CNN Architecture:** Built with PyTorch, featuring Convolutional Layers, Batch Normalization, Max Pooling, and Dropout for robust learning.
* **Data Augmentation:** Uses rotation, scaling, and translation during training to bridge the "domain shift" between perfect MNIST digits and messy human handwriting.
* **Smart Image Processing:** The backend automatically calculates the bounding box of user drawings, crops, centers, and pads them before prediction, perfectly mimicking the original MNIST dataset format.
* **Interactive Frontend:** A beautiful, responsive web UI built with HTML Canvas and Tailwind CSS. Supports both mouse and mobile touch inputs seamlessly.
* **REST API Backend:** A lightweight Flask server bridging the browser and the PyTorch model without requiring a page reload.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Backend:** Python, Flask, Flask-CORS
* **Frontend:** HTML5, JavaScript (Canvas API), Tailwind CSS
* **Image Processing:** Pillow (PIL)

## 📁 Project Structure

```text
Digit_recognize-Hand-written/
│
├── mnist_cnn.py         # Training script (defines CNN, downloads data, trains, and saves model)
├── mnist_cnn.pth        # Saved PyTorch model weights (generated after training)
├── server.py            # Flask backend server handling predictions
├── index.html           # Web UI for drawing digits
└── README.md            # Project documentation
```

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ installed. You will also need to install the required Python libraries:

```bash
pip install torch torchvision flask flask-cors pillow
```

*(Note for Mac users: If you get SSL errors during dataset download, ensure you have run the `Install Certificates.command` in your Mac's Python Application folder).*

### 2. Train the Model
First, you need to train the neural network and generate the `.pth` weight file. Run the training script:

```bash
python mnist_cnn.py
```
* This will download the MNIST dataset into a `data/` folder.
* It will train the model for the specified number of epochs.
* Once finished, it will save `mnist_cnn.pth` to your directory.

### 3. Start the Backend Server
Start the Flask server to load the trained model into memory and listen for requests from the web app:

```bash
python server.py
```
* The server will run locally on `http://127.0.0.1:5000`. Leave this terminal window open.

### 4. Open the Web App
Simply double-click the `index.html` file to open it in your default web browser. 

Draw a number in the canvas box and click **Predict Digit**. The browser will send your drawing to the Flask server, process it through the PyTorch model, and display the result instantly!

## 🧠 How the Accuracy was Improved
Standard MNIST models often fail on real-world web canvas drawings. This project solves that via two methods:
1. **Frontend-to-Backend Normalization:** The HTML canvas uses black ink on a white background. The Flask server uses `ImageOps.invert()` to swap this to white ink on a black background (matching MNIST). It then isolates the drawing using a bounding box (`getbbox()`), crops it, and centers it.
2. **Model Robustness:** By introducing `RandomRotation` and `RandomAffine` transforms during PyTorch training, the model learns to recognize tilted and off-center handwriting. `BatchNorm2d` and `Dropout(0.5)` were added to prevent overfitting.