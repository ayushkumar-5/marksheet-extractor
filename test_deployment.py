#!/usr/bin/env python3
"""
Test script to verify deployment configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
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
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_tesseract():
    """Test if Tesseract is available"""
    print("🔍 Testing Tesseract...")
    
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract is available")
            return True
        else:
            print("❌ Tesseract not working")
            return False
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def test_env_vars():
    """Test environment variables"""
    print("🔍 Testing environment variables...")
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "default_key":
        print("✅ GEMINI_API_KEY is set")
        return True
    else:
        print("⚠️  GEMINI_API_KEY not set or using default")
        return False

def test_files():
    """Test if required files exist"""
    print("🔍 Testing required files...")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "Dockerfile",
        "railway.json",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """Run all tests"""
    print("🚀 Testing deployment configuration...\n")
    
    tests = [
        ("File Check", test_files),
        ("Import Test", test_imports),
        ("Tesseract Test", test_tesseract),
        ("Environment Test", test_env_vars),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
