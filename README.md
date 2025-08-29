# 🎓 AI Marksheet Extraction API

An AI-powered service built with **FastAPI** that extracts structured data from academic marksheets (images or PDFs). It uses **OCR + LLM** to pull out candidate info, subjects, marks, and results — with **confidence scores** for each field.

👉 **Live Demo**: [Click here](https://marksheet-extractor-production.up.railway.app/demo)


This project was designed to handle **real-world, messy marksheets** in different formats and make the data machine-readable.

---

## ✨ Features

- 📂 **Multi-format input** → JPG, PNG, PDF (up to 10MB)
- 🔍 **Smart OCR** → Tesseract with OpenCV preprocessing + fallback OCR.space API
- 🤖 **AI-powered parsing** → Gemini LLM structures unstructured OCR text into clean JSON or HuggingFace(zephyr-7b-beta) if Gemini Fails
- 📊 **Confidence scores** → Every field includes a reliability score (0–1)
- 🖼️ **Bounding boxes** → Optional word-level bounding boxes for highlighting text regions
- 🎨 **Demo UI** → Upload a marksheet and view results in a neat, color-coded table
- 🛠 **Error handling** → File validation, OCR/LLM retries, clear error messages
- ⚡ **Production ready** → FastAPI backend, auto-docs, scalable deployment

---

## 🧮 Confidence Scoring Explained

The API provides **field-level confidence scores** (as required) and also a **global average OCR confidence** (extra metric for document quality).

### 1. Field-Level Confidence

Each field gets its own score using:

```
Final Field Confidence = (OCR Confidence × 0.4) + (LLM Confidence × 0.6)
```

- **OCR Confidence** → from Tesseract per word/line. For multi-word fields, values are averaged (or minimum for safety).
- **LLM Confidence** → Gemini evaluates certainty based on context and semantic clarity.

➡️ **Example:**
- Roll No → OCR 0.95, LLM 0.90 → Final = 0.93
- Mother's Name → OCR 0.65, LLM 0.75 → Final = 0.71

### 2. Global Average Confidence (Extra)

For additional monitoring, the API also computes an overall OCR quality score:

```python
avg_ocr_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
```

- This summarizes how clear the document was overall.
- It's not used directly in field scoring, but useful for debugging and quality checks.

### 3. Example Output

```json
{
  "candidate": {
    "name": {"value": "Student Name", "confidence": 0.95, "bbox": {...}},
    "roll_no": {"value": "123456", "confidence": 0.89, "bbox": {...}},
    "exam_year": {"value": "2024", "confidence": 0.92, "bbox": {...}}
  },
  "subjects": [
    {
      "subject": {"value": "Mathematics", "confidence": 0.98, "bbox": {...}},
      "max_marks": {"value": 100, "confidence": 0.85, "bbox": {...}},
      "obtained_marks": {"value": 85, "confidence": 0.87, "bbox": {...}},
      "grade": {"value": "A", "confidence": 0.91, "bbox": {...}}
    }
  ],
  "overall": {
    "result": {"value": "PASS", "confidence": 0.94, "bbox": {...}},
    "percentage": {"value": "85%", "confidence": 0.88, "bbox": {...}}
  }
}
```

---

## 🚀 API Endpoints

### `POST /extract`
Upload a marksheet (JPG, PNG, PDF). Returns structured JSON with field-level confidence scores.

### `GET /demo`
Interactive web UI for testing the API.
- Shows candidate info, subjects, results in tables
- Confidence shown as:
  - 🟢 Green → High (70%+)
  - 🟡 Yellow → Medium (50–70%)
  - 🔴 Red → Low (<50%)
- Allows downloading the full JSON output

---

## ⚙️ Installation & Setup

### System Dependencies

```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### Python Dependencies

```bash
uv add fastapi uvicorn pytesseract opencv-python pdf2image pillow \
   python-multipart requests google-genai jinja2
```

### Environment Variables

Create `.env` file:

```env

GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Run the Server

```bash
python main.py
```

Service will be available at:
- **API** → `http://localhost:5000`
- **Demo UI** → `http://localhost:5000/demo`
- **Docs** → `http://localhost:5000/docs`

---

## 🏗️ Architecture & Pipeline

1. **File validation** → Check type, size, format
2. **Image preprocessing** → OpenCV (grayscale, thresholding, denoising)
3. **Dual OCR** → Tesseract (primary) + OCR.space (fallback)
4. **AI parsing** → Gemini LLM cleans and structures extracted text
5. **Confidence scoring** → Field-level weighted formula + global average OCR
6. **Response** → Return JSON with `value`, `confidence`, and optional `bbox`

---

## Innovation / Extra Features

1. **Lightweight Sprint Board** – Built with Next.js 15 + App Router ensuring fast rendering.

2. **Animated UI** – Used Framer Motion for smooth drag & drop and transitions.

3. **Mock Auth & Guard** – Token-based login system (localStorage) with protected routes.

4. **Dynamic Task Management** – Tasks fetched from a mock API with state sync.

5. **Responsive & Modern Design** – Styled with Tailwind CSS for a clean experience.

## 🛡️ Error Handling

- ❌ **Invalid file** → returns 400
- ❌ **Too large (>10MB)** → returns 413
- ❌ **OCR/LLM failure** → retries, then 500 with error message

---

## ⚡ Performance & Reliability

- FastAPI ensures concurrency and speed
- Dual-OCR + preprocessing → higher accuracy on noisy scans
- Confidence scores make results **trustworthy & transparent**
- Demo UI provides real-time feedback and visualization

---


👨‍💻 Built for robustness, clarity, and generalizability — this API is ready to handle unseen marksheet formats while staying simple to run and test.
