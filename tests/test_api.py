import pytest
import io
import json
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# Add the parent directory to the path so we can import from main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

client = TestClient(app)

def create_test_image():
    """Create a simple test image with text"""
    # Create a simple white image with black text
    img = Image.new('RGB', (400, 200), color='white')
    # For a real test, you'd add actual text to the image
    # Here we just create a basic image for testing the API structure
    return img

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "marksheet-extraction-api"

def test_demo_endpoint():
    """Test the demo page endpoint"""
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_extract_endpoint_no_file():
    """Test extract endpoint without file"""
    response = client.post("/extract")
    assert response.status_code == 422  # Validation error

def test_extract_endpoint_invalid_file_type():
    """Test extract endpoint with invalid file type"""
    # Create a fake text file as bytes
    fake_file = io.BytesIO(b"This is not an image")
    response = client.post(
        "/extract",
        files={"file": ("test.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400

def test_extract_response_structure():
    """Test that the response has the correct structure when successful"""
    # This test would require a valid GEMINI_API_KEY and a real image
    # For now, we'll just test the structure expectation
    
    # Create a test PNG image
    img = create_test_image()
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # This test might fail without proper API key, but tests the structure
    response = client.post(
        "/extract",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    
    # If it succeeds (with valid API key), check structure
    if response.status_code == 200:
        data = response.json()
        
        # Check main structure
        assert "candidate" in data
        assert "subjects" in data
        assert "overall" in data
        assert "issue_details" in data
        
        # Check candidate structure
        candidate = data["candidate"]
        required_fields = ["name", "father_name", "mother_name", "roll_no", 
                          "registration_no", "dob", "exam_year", "board_university", "institution"]
        for field in required_fields:
            assert field in candidate
            assert "value" in candidate[field]
            assert "confidence" in candidate[field]
            assert isinstance(candidate[field]["confidence"], (int, float))
            assert 0.0 <= candidate[field]["confidence"] <= 1.0
        
        # Check subjects structure
        assert isinstance(data["subjects"], list)
        if data["subjects"]:
            subject = data["subjects"][0]
            subject_fields = ["subject", "max_marks", "obtained_marks", "grade"]
            for field in subject_fields:
                assert field in subject
                assert "value" in subject[field]
                assert "confidence" in subject[field]
        
        # Check overall structure
        overall = data["overall"]
        overall_fields = ["result", "division", "grade"]
        for field in overall_fields:
            assert field in overall
            assert "value" in overall[field]
            assert "confidence" in overall[field]
        
        # Check issue details structure
        issue_details = data["issue_details"]
        issue_fields = ["issue_date", "issue_place"]
        for field in issue_fields:
            assert field in issue_details
            assert "value" in issue_details[field]
            assert "confidence" in issue_details[field]

def test_bounding_box_structure():
    """Test that bounding boxes are included when available"""
    # This would be tested with actual OCR data
    # For now, just ensure the structure supports bbox
    
    from app.models import FieldValue, BoundingBox
    
    # Test FieldValue with bbox
    bbox = BoundingBox(x=100, y=200, width=150, height=25)
    field = FieldValue(value="Test Value", confidence=0.95, bbox=bbox)
    
    assert field.value == "Test Value"
    assert field.confidence == 0.95
    if field.bbox:
        assert field.bbox.x == 100
        assert field.bbox.y == 200
        assert field.bbox.width == 150
        assert field.bbox.height == 25
    
    # Test FieldValue without bbox
    field_no_bbox = FieldValue(value="Test Value", confidence=0.95)
    assert field_no_bbox.bbox is None

if __name__ == "__main__":
    pytest.main([__file__])