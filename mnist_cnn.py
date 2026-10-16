import torch
import torch.nn as nn

# 3. Define the Convolutional Neural Network (CNN)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # First Convolutional Layer: 1 input channel (grayscale), 16 output channels/filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer: 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected (Dense) Layers
        # After two MaxPools, the 28x28 image is reduced to 7x7. 
        # Flattened size: 32 channels * 7 * 7 = 1568
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 10 outputs (digits 0-9)

    def forward(self, x):
        # Extract features
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the 2D feature maps into a 1D vector
        x = x.view(-1, 32 * 7 * 7) 
        
        # Classify the features
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
