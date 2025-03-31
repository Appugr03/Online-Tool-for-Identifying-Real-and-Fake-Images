from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import torch.nn as nn
import os
import io
from werkzeug.utils import secure_filename
import pillow_avif  # Enables AVIF support in PIL

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# Define Upload Folder
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "avif"}  # Allowed file types
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the model architecture
class GANFaceDetector(nn.Module):
    def __init__(self):
        super(GANFaceDetector, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512 * 8 * 8)
        x = self.fc_layers(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GANFaceDetector()
model.load_state_dict(torch.load("best_face_detector_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file uploaded. Please select an image.")
        return redirect(request.url)

    file = request.files["file"]
    
    if file.filename == "":
        flash("No selected file. Please upload an image.")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("Unsupported file format. Allowed formats: PNG, JPG, JPEG, AVIF.")
        return redirect(request.url)

    try:
        # Secure filename and save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Open and process the image
        image = Image.open(file_path).convert("RGB")  # Ensures compatibility
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # Get confidence score
            confidence_score = probabilities[0][predicted.item()].item() * 100
            result = "Fake" if predicted.item() == 1 else "Real"

        return redirect(url_for("result", filename=filename, result=result, confidence=f"{confidence_score:.2f}"))

    except UnidentifiedImageError:
        flash("Error: The uploaded file is not a valid image.")
        return redirect(request.url)

@app.route("/result")
def result():
    filename = request.args.get("filename")
    result = request.args.get("result")
    confidence = request.args.get("confidence")
    image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template("result.html", image_url=image_url, result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
