import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Configure Tesseract (adjust path if needed)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux path

def preprocess_image_with_opencv(image: Image.Image) -> np.ndarray:
    """
    Preprocess image using OpenCV for better OCR accuracy.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Try multiple preprocessing approaches for better OCR
        # Approach 1: Gaussian blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        thresh1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Approach 2: Direct OTSU thresholding 
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Approach 3: Enhanced contrast + adaptive threshold
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        thresh3 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
        )
        
        # Choose the best result (for now, use enhanced version)
        cleaned = thresh3
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Resize if image is very small (improve OCR accuracy)
        height, width = cleaned.shape
        if height < 300 or width < 300:
            scale_factor = max(300 / height, 300 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        # Fallback: return original image as grayscale
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return img_array

def extract_text_from_image(image: Image.Image) -> Tuple[str, float]:
    """
    Extract text from image using Tesseract OCR with preprocessing.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (extracted_text, average_confidence)
    """
    try:
        # Preprocess image with OpenCV
        processed_image = preprocess_image_with_opencv(image)
        
        # OCR configuration for better accuracy
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/:-() '
        
        # Extract text with confidence scores
        data = pytesseract.image_to_data(
            processed_image, 
            config=custom_config, 
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate average confidence
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Only include text with reasonable confidence
            if text and conf > 30:  # Threshold for minimum confidence
                text_parts.append(text)
                confidences.append(conf / 100.0)  # Convert to 0-1 scale
        
        # Combine text
        extracted_text = ' '.join(text_parts)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        logger.info(f"OCR extracted {len(text_parts)} text elements with avg confidence: {avg_confidence:.3f}")
        
        return extracted_text, avg_confidence
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        # Fallback: try simple OCR without preprocessing
        try:
            simple_text = pytesseract.image_to_string(image)
            return simple_text, 0.5  # Default confidence for fallback
        except:
            return "", 0.0

def extract_text_with_bounding_boxes(image: Image.Image) -> dict:
    """
    Extract text with bounding box coordinates (bonus feature).
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with text and bounding box information
    """
    try:
        processed_image = preprocess_image_with_opencv(image)
        
        # Get bounding box data
        data = pytesseract.image_to_data(
            processed_image, 
            output_type=pytesseract.Output.DICT
        )
        
        boxes_data = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 30:
                boxes_data.append({
                    'text': text,
                    'confidence': conf / 100.0,
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return {
            'text_elements': boxes_data,
            'full_text': ' '.join([item['text'] for item in boxes_data])
        }
        
    except Exception as e:
        logger.error(f"Bounding box extraction failed: {str(e)}")
        return {'text_elements': [], 'full_text': ''}
