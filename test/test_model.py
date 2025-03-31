import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add project root to path

import pytest
import torch
from app import model, transform  # Import from app.py
from PIL import Image

@pytest.fixture
def sample_image():
    """Creates a sample tensor from a dummy image"""
    image = Image.new("RGB", (128, 128), "white")  # Dummy image
    image_tensor = transform(image).unsqueeze(0)  # Apply transforms
    return image_tensor

def test_model_load():
    """Check if the model loads properly"""
    assert model is not None, "Model failed to load"

def test_model_prediction(sample_image):
    """Check if model can make a prediction"""
    model.eval()
    with torch.no_grad():
        output = model(sample_image)
    
    assert output.shape == (1, 2), "Model output shape is incorrect"
    assert torch.is_tensor(output), "Model output is not a tensor"
