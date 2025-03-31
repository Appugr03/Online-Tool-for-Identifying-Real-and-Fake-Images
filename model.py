import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class GANFaceDetector(nn.Module):
    """
    CNN model for detecting GAN-generated faces.
    Architecture: 4 convolutional blocks followed by 3 fully connected layers.
    """
    def __init__(self):
        super(GANFaceDetector, self).__init__()
        
        # Convolutional layers
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
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Output 2 classes (real, fake)
        )
        
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128).
        Returns:
            Output tensor of shape (batch_size, 2).
        """
        x = self.conv_layers(x)
        x = x.view(-1, 512 * 8 * 8)  # Flatten feature maps
        x = self.fc_layers(x)
        return x  # No softmax here (CrossEntropyLoss expects raw logits)

    def _initialize_weights(self):
        """
        Initialize model weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Image transformation functions
def get_transforms(train=True):
    """
    Get image transformations for training or validation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# Function to load trained model
def load_model(model_path, device=None):
    """
    Load a trained model from a file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GANFaceDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Improved Prediction Function
def predict_image(model, image_tensor, device=None):
    """
    Make a prediction for a single image.
    Returns:
        prediction: 'Fake' or 'Real'
        confidence: Confidence score as percentage.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax
        _, predicted = torch.max(outputs, 1)
        
        confidence = probabilities[0][predicted.item()].item() * 100
        prediction = "Fake" if predicted.item() == 1 else "Real"

        # Adjust threshold if needed (e.g., classify as Fake if confidence < 80%)
        if prediction == "Real" and confidence < 80:
            prediction = "Fake"

        return prediction, confidence

if __name__ == "__main__":
    # Example usage
    model = GANFaceDetector()
    print("Model architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
