from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class BoundingBox(BaseModel):
    """Model for bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int

class FieldValue(BaseModel):
    """Model for a field with value, confidence score, and optional bounding box"""
    value: Any
    confidence: float
    bbox: Optional[BoundingBox] = None

class CandidateInfo(BaseModel):
    """Model for candidate information"""
    name: FieldValue
    father_name: FieldValue
    mother_name: FieldValue
    roll_no: FieldValue
    registration_no: FieldValue
    dob: FieldValue
    exam_year: FieldValue
    board_university: FieldValue
    institution: FieldValue

class Subject(BaseModel):
    """Model for subject information"""
    subject: FieldValue
    max_marks: FieldValue
    obtained_marks: FieldValue
    grade: FieldValue

class OverallResult(BaseModel):
    """Model for overall result information"""
    result: FieldValue
    division: FieldValue
    grade: FieldValue

class IssueDetails(BaseModel):
    """Model for marksheet issue details"""
    issue_date: FieldValue
    issue_place: FieldValue

class MarksheetResponse(BaseModel):
    """Complete marksheet response model"""
    candidate: CandidateInfo
    subjects: List[Subject]
    overall: OverallResult
    issue_details: IssueDetails
    
    class Config:
        json_encoders = {
            # Custom encoders if needed
        }
