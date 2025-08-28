import json
import logging
import os
import time
import requests
from google import genai
from google.genai import types
from app.models import MarksheetResponse, CandidateInfo, Subject, OverallResult, IssueDetails, FieldValue, BoundingBox
from app.utils import calculate_combined_confidence
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "default_key"))

def create_dual_ocr_prompt(tesseract_text: str, ocr_space_text: str) -> str:
    """
    Create a prompt for dual-OCR reconciliation following the pasted instructions.
    """
    prompt = f"""### Task
You will be given:
1. Tesseract OCR output (raw text with potential noise).
2. OCR.space OCR output (booster, more robust for noisy scans).
3. Some values may appear differently in each OCR. Your job is to reconcile them into the most accurate structured JSON.

### Rules
- Prefer text values where **both OCR outputs agree** (highest confidence).
- If values differ:
  - Select the more plausible one based on common marksheet formats (e.g., "Roll No" followed by digits).
  - If both seem valid, choose OCR.space output (it usually handles noisy scans better).
- Normalize fields (dates → YYYY-MM-DD, marks → integers/floats, grades → standard form).
- Include **confidence scores** (0–1) for every field:
  - High (≥0.9): Both OCRs agree OR one OCR is very clear.
  - Medium (0.7–0.89): Extracted from one OCR but matches common marksheet patterns.
  - Low (<0.7): Uncertain, incomplete, or guessed.

### JSON Schema
Return the data strictly in this format:
{{
  "candidate": {{
    "name": {{"value": "string", "confidence": 0.0}},
    "father_name": {{"value": "string", "confidence": 0.0}},
    "mother_name": {{"value": "string", "confidence": 0.0}},
    "roll_no": {{"value": "string", "confidence": 0.0}},
    "registration_no": {{"value": "string", "confidence": 0.0}},
    "dob": {{"value": "string", "confidence": 0.0}},
    "exam_year": {{"value": "string", "confidence": 0.0}},
    "board_university": {{"value": "string", "confidence": 0.0}},
    "institution": {{"value": "string", "confidence": 0.0}}
  }},
  "subjects": [
    {{
      "subject": {{"value": "string", "confidence": 0.0}},
      "max_marks": {{"value": 100, "confidence": 0.0}},
      "obtained_marks": {{"value": 85, "confidence": 0.0}},
      "grade": {{"value": "string", "confidence": 0.0}}
    }}
  ],
  "overall": {{
    "result": {{"value": "string", "confidence": 0.0}},
    "division": {{"value": "string", "confidence": 0.0}},
    "grade": {{"value": "string", "confidence": 0.0}}
  }},
  "issue_details": {{
    "issue_date": {{"value": "string", "confidence": 0.0}},
    "issue_place": {{"value": "string", "confidence": 0.0}}
  }}
}}

### Input
Tesseract OCR:
{tesseract_text[:1500]}

OCR.space OCR (booster):
{ocr_space_text[:1500]}

### Output
Return only the JSON schema filled with extracted field-values and confidence scores."""
    return prompt

def create_extraction_prompt(extracted_text: str) -> str:
    """
    Create a detailed prompt for the LLM to extract structured data from marksheet text.
    """
    prompt = f"""
You are an expert at extracting structured data from academic marksheets and transcripts. 
Analyze the following OCR-extracted text from a marksheet and extract the required information.

OCR Text:
{extracted_text}

Extract the following information and return it as JSON:

Return ONLY valid JSON in this exact format:
{{
  "candidate": {{
    "name": {{"value": "string", "confidence": 0.0}},
    "father_name": {{"value": "string", "confidence": 0.0}},
    "mother_name": {{"value": "string", "confidence": 0.0}},
    "roll_no": {{"value": "string", "confidence": 0.0}},
    "registration_no": {{"value": "string", "confidence": 0.0}},
    "dob": {{"value": "YYYY-MM-DD", "confidence": 0.0}},
    "exam_year": {{"value": "string", "confidence": 0.0}},
    "board_university": {{"value": "string", "confidence": 0.0}},
    "institution": {{"value": "string", "confidence": 0.0}}
  }},
  "subjects": [
    {{
      "subject": {{"value": "string", "confidence": 0.0}},
      "max_marks": {{"value": 100, "confidence": 0.0}},
      "obtained_marks": {{"value": 85, "confidence": 0.0}},
      "grade": {{"value": "A", "confidence": 0.0}}
    }}
  ],
  "overall": {{
    "result": {{"value": "Pass", "confidence": 0.0}},
    "division": {{"value": "First", "confidence": 0.0}},
    "grade": {{"value": "A", "confidence": 0.0}}
  }},
  "issue_details": {{
    "issue_date": {{"value": "YYYY-MM-DD", "confidence": 0.0}},
    "issue_place": {{"value": "string", "confidence": 0.0}}
  }}
}}

Important guidelines:
- If a field is not found, use empty string "" as value and confidence 0.0
- For dates, try to convert to YYYY-MM-DD format
- For marks, convert to integers/floats when possible
- Be conservative with confidence scores - only use high scores when very certain
- Ensure all JSON is properly formatted and valid
"""
    return prompt

def parse_marksheet_data(extracted_text: str, ocr_confidence: float, bbox_data: Optional[List[Dict]] = None, image_path: Optional[str] = None) -> MarksheetResponse:
    """
    Parse extracted text using Gemini LLM to structure marksheet data.
    
    Args:
        extracted_text: Raw text from OCR
        ocr_confidence: Average OCR confidence score
        
    Returns:
        Structured MarksheetResponse object
    """
    try:
        if not extracted_text.strip():
            raise ValueError("No text extracted from the marksheet")
        
        logger.info(f"Processing OCR text (length: {len(extracted_text)}) with confidence: {ocr_confidence:.3f}")
        logger.info(f"OCR text preview: {extracted_text[:300]}...")
        
        # Try dual-OCR reconciliation first if image_path is available
        parsed_data = None
        if image_path:
            logger.info("Attempting dual-OCR approach")
            ocr_space_text = get_ocr_space_text(image_path)
            if ocr_space_text:
                parsed_data = try_dual_ocr_reconciliation(extracted_text, ocr_space_text)
        
        # Fallback to single OCR with Gemini
        if not parsed_data:
            logger.info("Trying single-OCR with Gemini")
            parsed_data = try_gemini_with_retries(extracted_text)
        
        if not parsed_data:
            logger.warning("Gemini failed, trying HuggingFace Zephyr as backup")
            parsed_data = try_huggingface_zephyr(extracted_text, image_path)
        
        if not parsed_data:
            logger.error("Both Gemini and HuggingFace failed")
            return create_fallback_response(extracted_text, ocr_confidence)
        
        # Enhance confidence scores by combining with OCR confidence
        enhanced_data = enhance_confidence_scores(parsed_data, ocr_confidence)
        
        # Convert to Pydantic models
        marksheet_response = convert_to_pydantic_models(enhanced_data, bbox_data)
        
        logger.info("Successfully parsed marksheet data with LLM")
        return marksheet_response
        
    except Exception as e:
        logger.error(f"LLM parsing failed: {str(e)}")
        # Return fallback response with low confidence
        return create_fallback_response(extracted_text, ocr_confidence)

def try_gemini_with_retries(extracted_text: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Try Gemini API with retry logic for transient failures.
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Gemini attempt {attempt + 1}/{max_retries}")
            
            # Create prompt
            prompt = create_extraction_prompt(extracted_text)
            
            # Call Gemini API
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,  # Low temperature for consistent results
                    max_output_tokens=4000
                ),
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")
            
            # Parse JSON response
            raw_json = response.text.strip()
            logger.info(f"Gemini response length: {len(raw_json)} characters")
            
            parsed_data = json.loads(raw_json)
            logger.info("Successfully parsed Gemini response")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Gemini JSON decode error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Wait before retry
            
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)  # Wait longer for API errors
    
    return None

def try_huggingface_zephyr(extracted_text: str, image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Try multiple free backup methods when Gemini fails.
    """
    try:
        # 1. Try OCR.space free API (500/day, better OCR than Tesseract)
        if image_path:
            logger.info("Trying OCR.space free API for better text extraction")
            result = try_ocr_space_api(extracted_text, image_path)
            if result:
                return result
        
        # 2. Try HuggingFace if we have API key
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if hf_api_key:
            logger.info("Trying HuggingFace API")
            result = try_huggingface_api(extracted_text, hf_api_key)
            if result:
                return result
        
        # 3. Enhanced rule-based extraction
        logger.info("Using enhanced rule-based extraction as final backup")
        return try_rule_based_extraction(extracted_text)
        
    except Exception as e:
        logger.error(f"All backup methods failed: {str(e)}")
        return try_rule_based_extraction(extracted_text)

def try_ocr_space_api(extracted_text: str, image_path: str) -> Optional[Dict[str, Any]]:
    """
    Try OCR.space API to get better text extraction and parse it.
    """
    try:
        # Get better OCR text from OCR.space
        ocr_space_text = get_ocr_space_text(image_path)
        
        if not ocr_space_text:
            logger.warning("No text extracted from OCR.space")
            return None
            
        # Try dual-OCR reconciliation if we have both texts
        if len(ocr_space_text) > len(extracted_text):
            logger.info("Using OCR.space text for better accuracy")
            return try_dual_ocr_reconciliation(extracted_text, ocr_space_text)
        else:
            logger.info("Tesseract text was better, skipping OCR.space")
            return None
            
    except Exception as e:
        logger.error(f"OCR.space API processing error: {str(e)}")
        return None

def get_ocr_space_text(image_path: str) -> Optional[str]:
    """
    Get text from OCR.space free API - 500 requests/day, no registration needed.
    Returns the raw text for dual-OCR reconciliation.
    """
    try:
        logger.info("Getting text from OCR.space free API")
        
        url = 'https://api.ocr.space/parse/image'
        
        # Use free API key (no registration needed)
        data = {
            'apikey': 'helloworld',  # Free tier key
            'language': 'eng',
            'isOverlayRequired': False,  # Just get text, not bounding boxes
            'OCREngine': 2,  # Engine 2 for better accuracy
            'isTable': True,  # Table recognition for marksheets
        }
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            result = response.json()
            
            if result.get('IsErroredOnProcessing') == False:
                text_annotations = result.get('ParsedResults', [])
                if text_annotations:
                    ocr_space_text = text_annotations[0].get('ParsedText', '')
                    logger.info(f"OCR.space extracted text length: {len(ocr_space_text)}")
                    return ocr_space_text
                    
        logger.warning(f"OCR.space failed: {response.status_code}")
        return None
        
    except Exception as e:
        logger.error(f"OCR.space API error: {str(e)}")
        return None

def try_dual_ocr_reconciliation(tesseract_text: str, ocr_space_text: str) -> Optional[Dict[str, Any]]:
    """
    Use Gemini to reconcile both Tesseract and OCR.space outputs according to the pasted instructions.
    """
    try:
        logger.info("Attempting dual-OCR reconciliation with Gemini")
        
        # Create dual-OCR prompt
        prompt = create_dual_ocr_prompt(tesseract_text, ocr_space_text)
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=prompt)])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # Low temperature for consistent results
                max_output_tokens=4000
            ),
        )
        
        if not response.text:
            raise ValueError("Empty response from Gemini")
        
        # Parse JSON response
        raw_json = response.text.strip()
        logger.info(f"Dual-OCR Gemini response length: {len(raw_json)} characters")
        
        parsed_data = json.loads(raw_json)
        logger.info("Successfully parsed dual-OCR reconciliation response")
        return parsed_data
        
    except Exception as e:
        logger.error(f"Dual-OCR reconciliation failed: {str(e)}")
        return None

def try_huggingface_api(extracted_text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Try HuggingFace Inference API with authentication.
    """
    try:
        # Try multiple models in order of preference
        models_to_try = [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "microsoft/DialoGPT-medium", 
            "google/flan-t5-large"
        ]
        
        for model in models_to_try:
            logger.info(f"Trying HuggingFace model: {model}")
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Create a simpler prompt that works better with instruction models
            prompt = f"""Extract academic data from this text. Return only JSON format:

Text: {extracted_text[:500]}

Required JSON format:
{{
  "candidate": {{
    "name": {{"value": "", "confidence": 0.5}},
    "roll_no": {{"value": "", "confidence": 0.5}},
    "exam_year": {{"value": "", "confidence": 0.5}}
  }},
  "overall": {{
    "result": {{"value": "", "confidence": 0.5}}
  }}
}}

JSON:"""

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 800,
                    "temperature": 0.2,
                    "return_full_text": False,
                    "do_sample": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            logger.info(f"HuggingFace response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"HuggingFace result type: {type(result)}")
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    logger.info(f"Generated text: {generated_text[:200]}...")
                    
                    # Try to extract JSON
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = generated_text[json_start:json_end]
                        try:
                            parsed_data = json.loads(json_str)
                            logger.info(f"Successfully parsed HuggingFace response from {model}")
                            return parsed_data
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error from {model}: {e}")
                            continue
                elif isinstance(result, dict) and "generated_text" in result:
                    generated_text = result["generated_text"]
                    logger.info(f"Generated text (dict): {generated_text[:200]}...")
                    # Similar JSON extraction logic
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = generated_text[json_start:json_end]
                        try:
                            parsed_data = json.loads(json_str)
                            logger.info(f"Successfully parsed HuggingFace response from {model}")
                            return parsed_data
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error from {model}: {e}")
                            continue
                            
            else:
                logger.warning(f"HuggingFace model {model} failed: {response.status_code} - {response.text[:100]}")
                continue
                
        logger.error("All HuggingFace models failed")
        return None
        
    except Exception as e:
        logger.error(f"HuggingFace API error: {str(e)}")
        return None

def try_rule_based_extraction(text: str) -> Optional[Dict[str, Any]]:
    """
    Smart rule-based extraction for marksheet data.
    """
    import re
    
    logger.info(f"Analyzing text: {text[:200]}...")
    
    # Initialize structured response
    response = {
        "candidate": {},
        "subjects": [],
        "overall": {},
        "issue_details": {}
    }
    
    # Helper function to create field
    def make_field(value: str, confidence: float = 0.6):
        return {"value": value, "confidence": confidence}
    
    text_lines = text.split('\n')
    text_lower = text.lower()
    
    # Enhanced pattern matching
    patterns = {
        'roll_no': [
            r'roll\s*no[:\.\s]*([0-9]+)',
            r'roll[:\.\s]*([0-9]+)',
            r'student\s*(?:id|number)[:\.\s]*([0-9]+)',
            r'enrollment\s*no[:\.\s]*([0-9]+)',
            r'(?:^|\s)([0-9]{6,})'  # Any 6+ digit number
        ],
        'registration_no': [
            r'registration\s*no[:\.\s]*([A-Z0-9]+)',
            r'reg[:\.\s]*no[:\.\s]*([A-Z0-9]+)',
            r'admit\s*card[:\.\s]*([A-Z0-9]+)',
        ],
        'name': [
            r'name[:\.\s]*([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
            r'student[:\.\s]*([A-Z][a-z]+(?: [A-Z][a-z]+)*)',
        ],
        'year': [
            r'(?:exam|year)[:\.\s]*(20[0-9]{2})',
            r'(20[0-9]{2})',
        ],
        'board': [
            r'(cbse|icse|state\s*board|ncert|board\s*of\s*[a-z]+)',
        ]
    }
    
    # Extract data using patterns
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                
                if field == 'roll_no':
                    response['candidate']['roll_no'] = make_field(value, 0.7)
                elif field == 'registration_no':
                    response['candidate']['registration_no'] = make_field(value.upper(), 0.6)
                elif field == 'name':
                    response['candidate']['name'] = make_field(value.title(), 0.5)
                elif field == 'year':
                    response['candidate']['exam_year'] = make_field(value, 0.6)
                elif field == 'board':
                    response['candidate']['board_university'] = make_field(value.upper(), 0.5)
                break
    
    # Look for result/grade information
    if any(word in text_lower for word in ['pass', 'passed', 'qualify']):
        response['overall']['result'] = make_field('Pass', 0.6)
    elif any(word in text_lower for word in ['fail', 'failed']):
        response['overall']['result'] = make_field('Fail', 0.6)
    
    # Look for grades/marks
    grade_patterns = [
        r'(?:grade|class)[:\.\s]*([A-F][+\-]?)',
        r'([A-F][+\-]?)\s*grade',
        r'total[:\.\s]*([0-9]+)',
        r'marks[:\.\s]*([0-9]+)',
    ]
    
    for pattern in grade_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if matches[0].isdigit():
                # It's marks
                marks = int(matches[0])
                if marks > 100:  # Probably total marks
                    continue
                if marks >= 90:
                    response['overall']['grade'] = make_field('A+', 0.5)
                elif marks >= 80:
                    response['overall']['grade'] = make_field('A', 0.5)
                elif marks >= 70:
                    response['overall']['grade'] = make_field('B', 0.5)
                elif marks >= 60:
                    response['overall']['grade'] = make_field('C', 0.5)
            else:
                # It's a letter grade
                response['overall']['grade'] = make_field(matches[0].upper(), 0.5)
            break
    
    # Fill empty fields
    fields_to_fill = [
        ('candidate', ['name', 'father_name', 'mother_name', 'roll_no', 'registration_no', 'dob', 'exam_year', 'board_university', 'institution']),
        ('overall', ['result', 'division', 'grade']),
        ('issue_details', ['issue_date', 'issue_place'])
    ]
    
    for section, field_list in fields_to_fill:
        for field in field_list:
            if field not in response[section]:
                response[section][field] = make_field('', 0.1)
    
    # Log what we found
    found_data = {}
    for section_name, section_data in response.items():
        if isinstance(section_data, dict):
            for k, v in section_data.items():
                if isinstance(v, dict) and v.get('value'):
                    found_data[k] = v['value']
    logger.info(f"Rule-based extraction found: {found_data}")
    
    return response

def enhance_confidence_scores(parsed_data: Dict[str, Any], ocr_confidence: float) -> Dict[str, Any]:
    """
    Enhance confidence scores by combining LLM confidence with OCR confidence.
    
    Args:
        parsed_data: Parsed data from LLM
        ocr_confidence: OCR confidence score
        
    Returns:
        Enhanced data with combined confidence scores
    """
    def update_field_confidence(field_data):
        if isinstance(field_data, dict) and 'confidence' in field_data:
            llm_confidence = field_data['confidence']
            combined_confidence = calculate_combined_confidence(ocr_confidence, llm_confidence)
            field_data['confidence'] = round(combined_confidence, 3)
        return field_data
    
    # Update candidate fields
    if 'candidate' in parsed_data:
        for key, value in parsed_data['candidate'].items():
            parsed_data['candidate'][key] = update_field_confidence(value)
    
    # Update subject fields
    if 'subjects' in parsed_data:
        for subject in parsed_data['subjects']:
            for key, value in subject.items():
                subject[key] = update_field_confidence(value)
    
    # Update overall fields
    if 'overall' in parsed_data:
        for key, value in parsed_data['overall'].items():
            parsed_data['overall'][key] = update_field_confidence(value)
    
    # Update issue details
    if 'issue_details' in parsed_data:
        for key, value in parsed_data['issue_details'].items():
            parsed_data['issue_details'][key] = update_field_confidence(value)
    
    return parsed_data

def convert_to_pydantic_models(parsed_data: Dict[str, Any], bbox_data: Optional[List[Dict]] = None) -> MarksheetResponse:
    """
    Convert parsed dictionary to Pydantic models.
    
    Args:
        parsed_data: Parsed data dictionary
        
    Returns:
        MarksheetResponse object
    """
    try:
        # Helper function to find bounding box for a value
        def find_bbox_for_value(value: str, bbox_data: Optional[list] = None) -> Optional[BoundingBox]:
            if not bbox_data or not value:
                return None
            
            # Find the best matching bounding box for this value
            for bbox_item in bbox_data:
                if str(value).lower() in bbox_item['text'].lower():
                    return BoundingBox(
                        x=bbox_item['bbox']['x'],
                        y=bbox_item['bbox']['y'],
                        width=bbox_item['bbox']['width'],
                        height=bbox_item['bbox']['height']
                    )
            return None

        # Helper function to create FieldValue objects
        def create_field_value(data: Dict[str, Any]) -> FieldValue:
            bbox = find_bbox_for_value(data.get('value', ''), bbox_data)
            return FieldValue(
                value=data.get('value', ''),
                confidence=data.get('confidence', 0.0),
                bbox=bbox
            )
        
        # Create candidate info
        candidate_data = parsed_data.get('candidate', {})
        candidate = CandidateInfo(
            name=create_field_value(candidate_data.get('name', {})),
            father_name=create_field_value(candidate_data.get('father_name', {})),
            mother_name=create_field_value(candidate_data.get('mother_name', {})),
            roll_no=create_field_value(candidate_data.get('roll_no', {})),
            registration_no=create_field_value(candidate_data.get('registration_no', {})),
            dob=create_field_value(candidate_data.get('dob', {})),
            exam_year=create_field_value(candidate_data.get('exam_year', {})),
            board_university=create_field_value(candidate_data.get('board_university', {})),
            institution=create_field_value(candidate_data.get('institution', {}))
        )
        
        # Create subjects
        subjects_data = parsed_data.get('subjects', [])
        subjects = []
        for subject_data in subjects_data:
            subject = Subject(
                subject=create_field_value(subject_data.get('subject', {})),
                max_marks=create_field_value(subject_data.get('max_marks', {})),
                obtained_marks=create_field_value(subject_data.get('obtained_marks', {})),
                grade=create_field_value(subject_data.get('grade', {}))
            )
            subjects.append(subject)
        
        # Create overall result
        overall_data = parsed_data.get('overall', {})
        overall = OverallResult(
            result=create_field_value(overall_data.get('result', {})),
            division=create_field_value(overall_data.get('division', {})),
            grade=create_field_value(overall_data.get('grade', {}))
        )
        
        # Create issue details
        issue_data = parsed_data.get('issue_details', {})
        issue_details = IssueDetails(
            issue_date=create_field_value(issue_data.get('issue_date', {})),
            issue_place=create_field_value(issue_data.get('issue_place', {}))
        )
        
        return MarksheetResponse(
            candidate=candidate,
            subjects=subjects,
            overall=overall,
            issue_details=issue_details
        )
        
    except Exception as e:
        logger.error(f"Error converting to Pydantic models: {str(e)}")
        raise ValueError(f"Failed to structure response data: {str(e)}")

def create_fallback_response(extracted_text: str, ocr_confidence: float) -> MarksheetResponse:
    """
    Create a fallback response when LLM parsing fails.
    Try to extract basic information using simple text patterns.
    
    Args:
        extracted_text: Raw OCR text
        ocr_confidence: OCR confidence score
        
    Returns:
        Fallback MarksheetResponse with extracted data or empty fields
    """
    logger.warning("Creating fallback response due to LLM parsing failure")
    logger.info(f"Attempting basic text extraction from: {extracted_text[:500]}...")
    
    # Try basic text extraction patterns
    basic_data = extract_basic_data_patterns(extracted_text)
    
    # Create field with extracted value or empty
    def create_field(value: str = "", conf: float = 0.3) -> FieldValue:
        return FieldValue(value=value, confidence=min(conf, ocr_confidence))
    
    return MarksheetResponse(
        candidate=CandidateInfo(
            name=create_field(basic_data.get('name', ''), 0.4),
            father_name=create_field(basic_data.get('father_name', ''), 0.3),
            mother_name=create_field(basic_data.get('mother_name', ''), 0.3),
            roll_no=create_field(basic_data.get('roll_no', ''), 0.4),
            registration_no=create_field(basic_data.get('registration_no', ''), 0.3),
            dob=create_field(basic_data.get('dob', ''), 0.3),
            exam_year=create_field(basic_data.get('exam_year', ''), 0.3),
            board_university=create_field(basic_data.get('board', ''), 0.3),
            institution=create_field(basic_data.get('institution', ''), 0.3)
        ),
        subjects=basic_data.get('subjects', []),
        overall=OverallResult(
            result=create_field(basic_data.get('result', ''), 0.3),
            division=create_field(basic_data.get('division', ''), 0.3),
            grade=create_field(basic_data.get('grade', ''), 0.3)
        ),
        issue_details=IssueDetails(
            issue_date=create_field(basic_data.get('issue_date', ''), 0.2),
            issue_place=create_field(basic_data.get('issue_place', ''), 0.2)
        )
    )

def extract_basic_data_patterns(text: str) -> Dict[str, Any]:
    """
    Extract basic data using simple regex patterns as last resort.
    """
    import re
    
    data = {}
    text_lower = text.lower()
    
    try:
        # Look for roll number patterns
        roll_patterns = [
            r'roll\s*no[:\s]*([0-9]+)',
            r'roll[:\s]*([0-9]+)',
            r'student\s*id[:\s]*([0-9]+)',
        ]
        for pattern in roll_patterns:
            match = re.search(pattern, text_lower)
            if match:
                data['roll_no'] = match.group(1)
                break
        
        # Look for registration number
        reg_patterns = [
            r'registration\s*no[:\s]*([A-Z0-9]+)',
            r'reg[:\s]*no[:\s]*([A-Z0-9]+)',
        ]
        for pattern in reg_patterns:
            match = re.search(pattern, text_lower)
            if match:
                data['registration_no'] = match.group(1).upper()
                break
        
        # Look for year patterns
        year_patterns = [
            r'20[0-9]{2}',
            r'exam.*year[:\s]*([0-9]{4})',
        ]
        for pattern in year_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the most recent year
                data['exam_year'] = max(matches)
                break
        
        # Look for common result terms
        if any(word in text_lower for word in ['pass', 'passed']):
            data['result'] = 'Pass'
        elif any(word in text_lower for word in ['fail', 'failed']):
            data['result'] = 'Fail'
        
        # Look for division/grade patterns
        if 'first' in text_lower and 'division' in text_lower:
            data['division'] = 'First Division'
        elif 'second' in text_lower and 'division' in text_lower:
            data['division'] = 'Second Division'
        
        # Look for common boards
        boards = ['cbse', 'icse', 'state board', 'ncert']
        for board in boards:
            if board in text_lower:
                data['board'] = board.upper()
                break
        
        logger.info(f"Basic extraction found: {data}")
        return data
        
    except Exception as e:
        logger.error(f"Basic extraction failed: {e}")
        return data
