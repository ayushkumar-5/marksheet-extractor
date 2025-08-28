import os
import io
import tempfile
from fastapi import UploadFile, HTTPException
from PIL import Image
import pdf2image
from typing import List

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed file types
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf'}
ALLOWED_MIME_TYPES = {
    'image/jpeg', 
    'image/png', 
    'application/pdf'
}

def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file for size and type restrictions.
    
    Args:
        file: FastAPI UploadFile object
        
    Raises:
        ValueError: If file validation fails
    """
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE / (1024*1024):.1f}MB")
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"File type '{file.content_type}' not supported. Allowed types: JPG, PNG, PDF")
    
    # Check file extension
    if file.filename:
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File extension '{file_ext}' not supported. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")

def convert_pdf_to_images(pdf_content: bytes) -> List[Image.Image]:
    """
    Convert PDF content to PIL Images.
    
    Args:
        pdf_content: PDF file content as bytes
        
    Returns:
        List of PIL Images, one for each page
        
    Raises:
        Exception: If PDF conversion fails
    """
    try:
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_content)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                temp_pdf_path,
                dpi=300,  # High DPI for better OCR accuracy
                fmt='PNG',
                thread_count=1,
                use_cropbox=False,
                strict=False
            )
            
            return images
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_pdf_path)
            except:
                pass  # Ignore cleanup errors
                
    except Exception as e:
        raise Exception(f"Failed to convert PDF to images: {str(e)}")

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed PIL Image
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if image is too large (for processing efficiency)
        max_dimension = 3000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_width = int(image.size[0] * ratio)
            new_height = int(image.size[1] * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {str(e)}")

def calculate_combined_confidence(ocr_confidence: float, llm_confidence: float) -> float:
    """
    Calculate combined confidence score from OCR and LLM confidences.
    
    Args:
        ocr_confidence: OCR confidence score (0-1)
        llm_confidence: LLM confidence score (0-1)
        
    Returns:
        Combined confidence score (0-1)
    """
    # Weighted average: OCR confidence is more reliable for text presence,
    # LLM confidence is more reliable for semantic correctness
    return (ocr_confidence * 0.4 + llm_confidence * 0.6)
