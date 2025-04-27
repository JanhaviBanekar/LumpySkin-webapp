from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device("cpu")

class Discriminator(nn.Module):
    def __init__(self, num_classes=2):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = Discriminator(num_classes=2)
model.load_state_dict(torch.load("model/classifier_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, dim=1).item()
    return predicted

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction = predict_image(filepath)

    if prediction == 1:  # 1 = normal
        label = "✅ Prediction: NORMAL"
        recommendation = "📝 The skin appears healthy with no visible signs of Lumpy Skin Disease (LSD). No immediate veterinary attention is required. Continue to monitor the animal regularly and maintain good hygiene and nutrition."
    else:  # 0 = infected
        label = "⚠️ Prediction: INFECTED"
        recommendation = "📝 The image suggests signs of Lumpy Skin Disease (LSD). It is strongly advised to consult a veterinarian immediately. Isolate the animal from the herd to prevent further spread. Maintain hygiene, and avoid direct contact without protective measures."
        
    # Add model accuracy
    model_accuracy = "📊 Model Accuracy: 97.34%"

    return render_template('result.html',
                           label=label,
                           recommendation=recommendation,
                           image_url=url_for('static', filename=f'uploads/{filename}'),accuracy=model_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
