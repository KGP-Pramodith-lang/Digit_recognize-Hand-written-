from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import base64
import io

# 1. Define Model Architecture (Needed to load the weights)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Initialize Flask App
app = Flask(__name__)
# CORS allows your browser's HTML file to securely talk to this Python server
CORS(app) 

# 3. Load PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)

try:
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    model.eval() # Set to evaluation mode
    print("--> PyTorch Model loaded successfully! Ready for predictions.")
except FileNotFoundError:
    print("--> ERROR: 'mnist_cnn.pth' not found. Ensure it is in the same folder.")

# 4. Define Image Pre-processing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 5. Create the API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Strip the standard HTML canvas "data:image/png;base64," prefix
        image_data = data['image'].split(',')[1]
        
        # Convert base64 string back to binary image bytes
        image_bytes = base64.b64decode(image_data)
        
        # Open as PIL Image and convert to grayscale ('L')
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Invert colors: HTML Canvas is black ink on white, MNIST expects white on black
        inverted_image = ImageOps.invert(image)
        
        # Apply transforms and add batch dimension (1 image)
        tensor_image = transform(inverted_image).unsqueeze(0).to(device)
        
        # Make the prediction
        with torch.no_grad():
            output = model(tensor_image)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
            
        # Send the answer back to the HTML page
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)