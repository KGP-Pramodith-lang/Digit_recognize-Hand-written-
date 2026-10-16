from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import base64
import io

app = Flask(__name__)
CORS(app) # Allows the HTML page to communicate with this server

# 1. Define the Updated CNN Architecture
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) 
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# 2. Load the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)

try:
    # Make sure you re-trained and saved a new mnist_cnn.pth after adding BatchNorm/Dropout!
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'mnist_cnn.pth' not found. Please ensure it is in the same folder.")

# 3. Define the transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 4. Create the API Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 image string from the frontend
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        
        # Open image and convert to grayscale
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        
        # Invert colors (MNIST is white ink on black background)
        img = ImageOps.invert(img)

        # --- NEW CROPPING LOGIC STARTS HERE ---
        # Find the bounding box of the non-black pixels (the drawing)
        bbox = img.getbbox()
        if bbox:
            # Crop the image to just the drawing
            img = img.crop(bbox)
            # Add a little black padding around it so it doesn't touch the edges
            img = ImageOps.expand(img, border=20, fill=0)
        # --- NEW CROPPING LOGIC ENDS HERE ---

        # Apply transformations and add batch dimension
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Runs the server on port 5000
    app.run(port=5000, debug=True)