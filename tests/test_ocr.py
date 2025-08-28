import pytest
from PIL import Image
import numpy as np
from app.ocr import preprocess_image_with_opencv, extract_text_from_image, extract_text_with_bounding_boxes

def create_test_image_with_text():
    """Create a test image with some text for OCR testing"""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    # In a real scenario, you'd add text using PIL ImageDraw
    # For testing purposes, we'll use a simple white image
    return img

def test_preprocess_image_with_opencv():
    """Test OpenCV preprocessing function"""
    img = create_test_image_with_text()
    processed = preprocess_image_with_opencv(img)
    
    # Check that the output is a numpy array
    assert isinstance(processed, np.ndarray)
    # Check that it's grayscale (2D array)
    assert len(processed.shape) == 2
    # Check that dimensions are reasonable
    assert processed.shape[0] > 0 and processed.shape[1] > 0

def test_extract_text_from_image():
    """Test text extraction from image"""
    img = create_test_image_with_text()
    text, confidence = extract_text_from_image(img)
    
    # Check return types
    assert isinstance(text, str)
    assert isinstance(confidence, (int, float))
    # Check confidence is in valid range
    assert 0.0 <= confidence <= 1.0

def test_extract_text_with_bounding_boxes():
    """Test bounding box extraction"""
    img = create_test_image_with_text()
    result = extract_text_with_bounding_boxes(img)
    
    # Check structure
    assert isinstance(result, dict)
    assert 'text_elements' in result
    assert 'full_text' in result
    assert isinstance(result['text_elements'], list)
    assert isinstance(result['full_text'], str)
    
    # If there are text elements, check their structure
    for element in result['text_elements']:
        assert 'text' in element
        assert 'confidence' in element
        assert 'bbox' in element
        assert isinstance(element['confidence'], (int, float))
        assert 0.0 <= element['confidence'] <= 1.0
        
        bbox = element['bbox']
        assert 'x' in bbox
        assert 'y' in bbox
        assert 'width' in bbox
        assert 'height' in bbox
        assert all(isinstance(bbox[key], int) for key in ['x', 'y', 'width', 'height'])

def test_image_preprocessing_edge_cases():
    """Test edge cases for image preprocessing"""
    # Test very small image
    small_img = Image.new('RGB', (50, 30), color='white')
    processed = preprocess_image_with_opencv(small_img)
    # Should be resized to at least 300px
    assert max(processed.shape) >= 300
    
    # Test grayscale image
    gray_img = Image.new('L', (400, 200), color=255)
    processed_gray = preprocess_image_with_opencv(gray_img)
    assert isinstance(processed_gray, np.ndarray)
    assert len(processed_gray.shape) == 2

if __name__ == "__main__":
    pytest.main([__file__])