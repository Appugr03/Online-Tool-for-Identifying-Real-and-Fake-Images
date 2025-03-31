import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app  # Import your Flask app
import pytest

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True
    client = app.test_client()
    yield client

def test_homepage(client):
    """Test if the homepage loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Fake Image Detection" in response.data  # Adjust based on index.html content

def test_file_upload(client):
    """Test the image upload functionality."""
    test_image_path = "static/uploads/test_image.jpg"  # Ensure this file exists

    # Ensure the file exists before running the test
    assert os.path.exists(test_image_path), "Test image not found. Place a sample image in static/uploads."

    with open(test_image_path, "rb") as img:
        data = {"file": (img, "test_image.jpg")}
        response = client.post("/upload", data=data, content_type="multipart/form-data")
    
    assert response.status_code == 200
    assert b"Upload Successful" in response.data  # Adjust based on actual upload response
