# ATS Score Calculator API

A production-ready FastAPI application for calculating ATS (Applicant Tracking System) compatibility scores for resumes. Supports both standalone resume analysis and job description matching.

## Features

- **Dual Mode Scoring**:
  - With Job Description: Compares resume against JD
  - Standalone: Evaluates resume quality independently
  
- **File Support**: PDF, DOCX, TXT
- **Multi-column Resume Handling**: Uses pdfplumber for better text extraction
- **Semantic Similarity**: Sentence Transformers + TF-IDF fallback
- **Text Splitting**: Handles long documents with chunking
- **Docker Support**: Ready for containerized deployment

## Quick Start

### Local Development

1. **Clone and setup**:
```bash
git clone <your-repo-url>
cd ats-score-calculator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the API**:
```bash
uvicorn app.main:app --reload
```

4. **Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Docker

```bash
docker-compose up -d
```

## API Endpoints

### Calculate ATS Score (Text)
```bash
POST /api/v1/ats-score
Content-Type: application/json

{
  "resume_text": "Your resume text here...",
  "jd_text": "Optional job description..."
}
```

### Calculate ATS Score (File Upload)
```bash
POST /api/v1/ats-score/upload
Content-Type: multipart/form-data

resume_file: <PDF/DOCX/TXT file>
jd_text: "Optional job description"
```

### Health Check
```bash
GET /health
```

## Scoring Breakdown

### With Job Description
- Keyword Match: 20%
- Skills Match: 25%
- Semantic Similarity: 15%
- Experience Match: 15%
- Education Match: 10%
- Formatting: 10%
- Completeness: 5%

### Standalone Mode
- Skills Density: 25%
- Experience: 20%
- Education: 15%
- Formatting: 25%
- Completeness: 15%

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
ATS_DEBUG=false
ATS_USE_EMBEDDINGS=true
ATS_MAX_FILE_SIZE_MB=10
```

## Requirements

- Python 3.11+
- FastAPI
- sentence-transformers (optional, for embeddings)
- PyTorch (optional, for embeddings)

## License

MIT
