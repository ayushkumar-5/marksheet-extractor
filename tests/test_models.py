import pytest
from app.models import FieldValue, BoundingBox, CandidateInfo, Subject, OverallResult, IssueDetails, MarksheetResponse

def test_bounding_box_model():
    """Test BoundingBox model"""
    bbox = BoundingBox(x=100, y=200, width=150, height=25)
    assert bbox.x == 100
    assert bbox.y == 200
    assert bbox.width == 150
    assert bbox.height == 25

def test_field_value_model():
    """Test FieldValue model with and without bounding box"""
    # With bounding box
    bbox = BoundingBox(x=100, y=200, width=150, height=25)
    field_with_bbox = FieldValue(value="Test Value", confidence=0.95, bbox=bbox)
    assert field_with_bbox.value == "Test Value"
    assert field_with_bbox.confidence == 0.95
    assert field_with_bbox.bbox is not None
    assert field_with_bbox.bbox.x == 100
    
    # Without bounding box
    field_without_bbox = FieldValue(value="Test Value", confidence=0.85)
    assert field_without_bbox.value == "Test Value"
    assert field_without_bbox.confidence == 0.85
    assert field_without_bbox.bbox is None

def test_candidate_info_model():
    """Test CandidateInfo model"""
    name = FieldValue(value="John Doe", confidence=0.95)
    father_name = FieldValue(value="Robert Doe", confidence=0.90)
    mother_name = FieldValue(value="Jane Doe", confidence=0.88)
    roll_no = FieldValue(value="123456", confidence=0.98)
    registration_no = FieldValue(value="REG789", confidence=0.92)
    dob = FieldValue(value="2000-01-15", confidence=0.85)
    exam_year = FieldValue(value="2022", confidence=0.95)
    board_university = FieldValue(value="CBSE", confidence=0.90)
    institution = FieldValue(value="Test School", confidence=0.93)
    
    candidate = CandidateInfo(
        name=name,
        father_name=father_name,
        mother_name=mother_name,
        roll_no=roll_no,
        registration_no=registration_no,
        dob=dob,
        exam_year=exam_year,
        board_university=board_university,
        institution=institution
    )
    
    assert candidate.name.value == "John Doe"
    assert candidate.roll_no.value == "123456"
    assert candidate.dob.value == "2000-01-15"

def test_subject_model():
    """Test Subject model"""
    subject = Subject(
        subject=FieldValue(value="Mathematics", confidence=0.95),
        max_marks=FieldValue(value=100, confidence=0.98),
        obtained_marks=FieldValue(value=85, confidence=0.92),
        grade=FieldValue(value="A", confidence=0.90)
    )
    
    assert subject.subject.value == "Mathematics"
    assert subject.max_marks.value == 100
    assert subject.obtained_marks.value == 85
    assert subject.grade.value == "A"

def test_overall_result_model():
    """Test OverallResult model"""
    overall = OverallResult(
        result=FieldValue(value="Pass", confidence=0.98),
        division=FieldValue(value="First", confidence=0.90),
        grade=FieldValue(value="A", confidence=0.88)
    )
    
    assert overall.result.value == "Pass"
    assert overall.division.value == "First"
    assert overall.grade.value == "A"

def test_issue_details_model():
    """Test IssueDetails model"""
    issue_details = IssueDetails(
        issue_date=FieldValue(value="2022-07-15", confidence=0.85),
        issue_place=FieldValue(value="Delhi", confidence=0.80)
    )
    
    assert issue_details.issue_date.value == "2022-07-15"
    assert issue_details.issue_place.value == "Delhi"

def test_complete_marksheet_response():
    """Test complete MarksheetResponse model"""
    # Create all components
    candidate = CandidateInfo(
        name=FieldValue(value="John Doe", confidence=0.95),
        father_name=FieldValue(value="Robert Doe", confidence=0.90),
        mother_name=FieldValue(value="Jane Doe", confidence=0.88),
        roll_no=FieldValue(value="123456", confidence=0.98),
        registration_no=FieldValue(value="REG789", confidence=0.92),
        dob=FieldValue(value="2000-01-15", confidence=0.85),
        exam_year=FieldValue(value="2022", confidence=0.95),
        board_university=FieldValue(value="CBSE", confidence=0.90),
        institution=FieldValue(value="Test School", confidence=0.93)
    )
    
    subjects = [
        Subject(
            subject=FieldValue(value="Mathematics", confidence=0.95),
            max_marks=FieldValue(value=100, confidence=0.98),
            obtained_marks=FieldValue(value=85, confidence=0.92),
            grade=FieldValue(value="A", confidence=0.90)
        ),
        Subject(
            subject=FieldValue(value="Science", confidence=0.93),
            max_marks=FieldValue(value=100, confidence=0.98),
            obtained_marks=FieldValue(value=78, confidence=0.89),
            grade=FieldValue(value="B+", confidence=0.87)
        )
    ]
    
    overall = OverallResult(
        result=FieldValue(value="Pass", confidence=0.98),
        division=FieldValue(value="First", confidence=0.90),
        grade=FieldValue(value="A", confidence=0.88)
    )
    
    issue_details = IssueDetails(
        issue_date=FieldValue(value="2022-07-15", confidence=0.85),
        issue_place=FieldValue(value="Delhi", confidence=0.80)
    )
    
    # Create complete response
    response = MarksheetResponse(
        candidate=candidate,
        subjects=subjects,
        overall=overall,
        issue_details=issue_details
    )
    
    assert response.candidate.name.value == "John Doe"
    assert len(response.subjects) == 2
    assert response.subjects[0].subject.value == "Mathematics"
    assert response.overall.result.value == "Pass"
    assert response.issue_details.issue_date.value == "2022-07-15"

if __name__ == "__main__":
    pytest.main([__file__])