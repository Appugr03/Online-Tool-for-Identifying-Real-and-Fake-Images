import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from model import GANFaceDetector, get_transforms
import matplotlib.pyplot as plt
import numpy as np

# Face Dataset Class
class FaceDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, transform=None):
        self.transform = transform
        self.data = []

        for real_dir in real_dirs:
            if not os.path.exists(real_dir):
                print(f"Warning: Real directory not found -> {real_dir}")
                continue
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append({'path': os.path.join(real_dir, img_name), 'label': 0})

        for fake_dir in fake_dirs:
            if not os.path.exists(fake_dir):
                print(f"Warning: Fake directory not found -> {fake_dir}")
                continue
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append({'path': os.path.join(fake_dir, img_name), 'label': 1})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        image = Image.open(img_data['path']).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        return image, img_data['label']

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Forward pass
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):  # Mixed precision for CPU
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100 * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Training Loss: {train_losses[-1]:.4f}, Accuracy: {train_accs[-1]:.2f}%")
        print(f"Validation Loss: {val_losses[-1]:.4f}, Accuracy: {val_accs[-1]:.2f}%")

        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            torch.save(model.state_dict(), 'best_face_detector_model.pth')
            print(f"Model saved with validation accuracy: {val_accs[-1]:.2f}%")
        print('-' * 60)

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# Main Function
def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    real_dirs = [r"C:\Users\apoor\Project\Dataset\real_and_fake_face\training_real"]
    fake_dirs = [r"C:\Users\apoor\Project\Dataset\real_and_fake_face\training_fake"]

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    full_dataset = FaceDataset(real_dirs, fake_dirs, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Optimize data loading for CPU
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    print(f"Total images: {len(full_dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    model = GANFaceDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

    print("Training completed!")
    print("Best model saved as 'best_face_detector_model.pth'")
    print("Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    main()
