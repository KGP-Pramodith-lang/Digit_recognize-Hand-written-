import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as transforms
# Import your model architecture from the first script
from mnist_cnn import MNIST_CNN 

class DigitPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw & Predict")
        
        # --- 1. Load the PyTorch Model ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNIST_CNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load('mnist_cnn.pth', map_location=self.device))
            self.model.eval()
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Error: 'mnist_cnn.pth' not found. Please train and save the model first.")
        
        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # --- 2. Set Up the UI ---
        self.canvas_width, self.canvas_height = 280, 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)

        # Invisible PIL image to capture the drawing
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind('<B1-Motion>', self.paint)
        
        # UI Elements: Predict Button, Clear Button, and Result Label
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=5)
        
        self.predict_btn = tk.Button(self.btn_frame, text="Predict Digit", command=self.predict_digit, bg="lightgreen")
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = tk.Button(self.btn_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.result_label = tk.Label(self.root, text="Draw a number and click Predict", font=('Arial', 14))
        self.result_label.pack(pady=10)

    def paint(self, event):
        brush_size = 15
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def predict_digit(self):
        # 1. Invert colors (MNIST is white ink on black background)
        inverted_image = ImageOps.invert(self.image)
        
        # 2. Apply transformations and add batch dimension
        img_tensor = self.transform(inverted_image).unsqueeze(0).to(self.device)
        
        # 3. Make prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
            
        # 4. Update the UI with the result
        self.result_label.config(text=f"Prediction: {prediction}", fg="blue", font=('Arial', 18, 'bold'))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a number and click Predict", fg="black", font=('Arial', 14))

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitPredictorApp(root)
    root.mainloop()