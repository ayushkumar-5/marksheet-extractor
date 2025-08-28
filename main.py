import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request

# Load environment variables from .env file
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.utils import validate_file, convert_pdf_to_images
from app.ocr import extract_text_from_image, extract_text_with_bounding_boxes
from app.llm_parser import parse_marksheet_data
from app.models import MarksheetResponse
import tempfile
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Marksheet Extraction API",
    description="Extract structured data from academic marksheets using OCR and LLM processing",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "AI Marksheet Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "extract": "/extract - POST endpoint for single marksheet extraction",
            "health": "/health - GET endpoint for health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "marksheet-extraction-api"}

@app.get("/demo")
async def demo_page(request: Request):
    """Demo page for testing the API"""
    return templates.TemplateResponse("demo.html", {"request": request})

@app.post("/extract", response_model=MarksheetResponse)
async def extract_marksheet_data(file: UploadFile = File(...)):
    """
    Extract structured data from academic marksheet image or PDF.
    
    Accepts JPG, PNG, or PDF files up to 10MB.
    Returns structured JSON with confidence scores for each field.
    """
    try:
        # Validate file
        validate_file(file)
        
        # Read file content
        file_content = await file.read()
        
        # Save uploaded file temporarily for backup OCR processing
        temp_file_path = None
        try:
            suffix = f".{file.filename.split('.')[-1]}" if file.filename and '.' in file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
        except Exception as e:
            logger.warning(f"Failed to save temp file: {e}")
        
        # Process based on file type
        if file.content_type == "application/pdf":
            # Convert PDF to images
            images = convert_pdf_to_images(file_content)
            
            # Extract text from all pages
            all_text = ""
            all_confidences = []
            all_bbox_data = []
            
            for image in images:
                text, confidence = extract_text_from_image(image)
                bbox_info = extract_text_with_bounding_boxes(image)
                all_text += f"\n{text}"
                all_confidences.append(confidence)
                all_bbox_data.extend(bbox_info['text_elements'])
            
            # Average confidence across pages
            avg_ocr_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
        else:
            # Handle image files (JPG, PNG)
            image = Image.open(io.BytesIO(file_content))
            all_text, avg_ocr_confidence = extract_text_from_image(image)
            bbox_info = extract_text_with_bounding_boxes(image)
            all_bbox_data = bbox_info['text_elements']
        
        # Parse extracted text using LLM
        structured_data = parse_marksheet_data(all_text, avg_ocr_confidence, all_bbox_data, temp_file_path)
        
        logger.info(f"Successfully processed file: {file.filename}")
        return structured_data
        
    except ValueError as e:
        logger.error(f"Validation error for file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Processing error for file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process marksheet: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred while processing your request"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
