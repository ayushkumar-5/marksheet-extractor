#!/bin/bash

# Build script for Railway deployment

echo "🚀 Building AI Marksheet Extraction API..."

# Check if we're in a Railway environment
if [ -n "$RAILWAY_ENVIRONMENT" ]; then
    echo "✅ Running in Railway environment"
else
    echo "ℹ️  Running locally"
fi

# Install system dependencies (if needed)
if command -v apt-get &> /dev/null; then
    echo "📦 Installing system dependencies..."
    apt-get update && apt-get install -y \
        tesseract-ocr \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Test Tesseract installation
echo "🔍 Testing Tesseract installation..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract is installed"
    tesseract --version
else
    echo "❌ Tesseract not found"
    exit 1
fi

# Test Python imports
echo "🐍 Testing Python imports..."
python3 -c "
import fastapi
import uvicorn
import pytesseract
import cv2
import pdf2image
from PIL import Image
import requests
import google.genai as genai
import jinja2
from dotenv import load_dotenv
print('✅ All imports successful')
"

echo "🎉 Build completed successfully!"
