# 📝 Approach Note – AI Marksheet Extraction API

## 1. Objective

The goal is to build an AI-powered marksheet extraction API that takes image/PDF files and produces structured JSON output with field-level confidence scores.

The solution must be generalizable, robust to unseen marksheets, and production-ready.

---

## 2. Architecture

**Pipeline Overview**

1. **File validation** → Check extension & size (≤10 MB)
2. **Preprocessing** → OpenCV (grayscale, threshold, noise removal)
3. **Dual OCR** →
   - Primary: Tesseract (open-source, good for text-heavy docs)
   - Fallback: OCR.space API (robust with noisy/complex layouts)
4. **LLM Parsing & Validation** →
   - Primary: Gemini API
   - Fallback: Hugging Face Zephyr
   - LLM ensures normalization (e.g., DOB → `YYYY-MM-DD`)
5. **Confidence Scoring** → Weighted formula:
   ```
   Final = 0.4 × OCR + 0.6 × LLM
   ```
6. **Schema Validation** → JSON output validated against Pydantic models
7. **Response** → Structured JSON `{ value, confidence, bbox }`

---

## 3. Extracted Fields

### **Candidate**
- Name, Father's Name, Mother's Name
- Roll No, Registration No
- Date of Birth
- Exam Year
- Board/University
- Institution

### **Subjects**
- Subject name
- Max Marks / Credits
- Obtained Marks / Credits
- Grade

### **Overall**
- Result
- Percentage
- Division (if present)

### **Metadata**
- Issue Date
- Issue Place

---

## 4. Confidence Scoring

Each field = weighted sum of OCR + LLM:
- **OCR confidence**: from Tesseract (word-level average)
- **LLM confidence**: estimated from semantic clarity & prompt probability

This combination ensures robustness:
- OCR handles clean scans
- LLM corrects noisy or context-dependent fields

---

## 5. Error Handling

- **Invalid file** → 400
- **Large file (>10MB)** → 413
- **OCR/LLM failure** → retries + fallback
- **Schema validation failure** → 500

---

## 6. Deployment

- **Framework**: FastAPI
- **Containerized** (Docker optional)
- **Free hosting**: Railway / Render
- **Auto-generated docs**: `/docs` (Swagger UI)

---

## 7. Innovations

- **Dual OCR (Tesseract + OCR.space)** → robust across formats
- **LLM Fallback (Gemini + Hugging Face)** → prevents downtime
- **Confidence Calibration** → weighted scoring, transparent
- **Demo UI** → visualization with color-coded confidence
- **Bounding Boxes** → supports text highlighting

---

## 8. Limitations & Future Work

- Handwritten marksheets are still challenging
- Confidence could be further calibrated using cross-field consistency checks
- Batch processing and authentication can be extended

---

## 9. Conclusion

This system provides a reliable, generalizable, and transparent way to extract structured marksheet data.

By combining **OCR, preprocessing, and LLM parsing**, it handles messy real-world inputs and produces machine-readable outputs with confidence scores.
