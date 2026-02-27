"""ATS Score Calculator - FastAPI Application."""

import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.file_parser import FileParseError, extract_text_from_file, get_supported_extensions
from app.models import ATSRequest, ATSResponse, ErrorResponse, HealthResponse
from app.scoring import (
    calculate_ats_score,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    init_embedding_model,  # ✅ preload embeddings
)

# ---------------------------
# Logging (avoid double-config with uvicorn --reload)
# ---------------------------
def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:  # uvicorn already configured logging
        return
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


setup_logging()
logger = logging.getLogger(__name__)

# Max file size from config
MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # ✅ Preload embedding model ONCE (prevents first-request freeze)
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.info("✅ Sentence Transformers (embeddings) is AVAILABLE")
        init_embedding_model()
    else:
        logger.warning("❌ Sentence Transformers (embeddings) is NOT available - similarity will use TF-IDF only")

    yield

    logger.info("Shutting down application")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-ready API for calculating ATS (Applicant Tracking System) scores. "
        "Accepts resume text or a resume file upload. Job description is optional."
    ),
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred", "error_code": "INTERNAL_ERROR"},
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Health Check")
async def health_check() -> HealthResponse:
    """Return health status of the API."""
    return HealthResponse(status="healthy", version=settings.app_version)


@app.post(
    "/api/v1/ats-score",
    response_model=ATSResponse,
    tags=["ATS Score"],
    summary="Calculate ATS Score (Text Input)",
    description="Calculate ATS score using resume_text and optional jd_text (best for resume builders).",
)
async def calculate_score(request: ATSRequest) -> ATSResponse:
    """Calculate ATS score using plain text input."""
    try:
        logger.info("Processing ATS score calculation request (text)")
        result = calculate_ats_score(resume_text=request.resume_text, jd_text=request.jd_text)
        logger.info(f"ATS score calculated: {result.get('overall_score')}")
        return ATSResponse(**result)

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.exception(f"Error calculating ATS score: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate ATS score")


@app.post(
    "/api/v1/ats-score/upload",
    response_model=ATSResponse,
    tags=["ATS Score"],
    summary="Calculate ATS Score (File Upload)",
    description="Calculate ATS score by uploading a resume file (PDF, DOCX, TXT) with optional JD text.",
)
async def calculate_score_upload(
    resume_file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    jd_text: Optional[str] = Form(default=None, description="Optional job description text"),
) -> ATSResponse:
    """Calculate ATS score by uploading a resume file."""
    try:
        logger.info(f"Processing file upload: {resume_file.filename}")

        if not resume_file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")

        supported = get_supported_extensions()
        if not any(resume_file.filename.lower().endswith(ext) for ext in supported):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported: {', '.join(supported)}",
            )

        content = await resume_file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        if len(content) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")

        file_obj = io.BytesIO(content)
        try:
            resume_text = extract_text_from_file(file_obj, resume_file.filename)
            logger.info(f"Extracted {len(resume_text)} characters from file")
        except FileParseError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        # ✅ NEW: config-driven extraction quality checks
        extracted_len = len(resume_text.strip())

        # warn when extraction looks weak (likely scanned PDF)
        if extracted_len < settings.low_text_chars_threshold:
            logger.warning(
                f"Low text extraction ({extracted_len} chars). File may be image-based or have extraction issues."
            )

        # hard reject only if below minimum
        if extracted_len < settings.min_extracted_chars:
            logger.warning(f"Extracted text too short: '{resume_text[:200]}...'")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Resume content too short ({extracted_len} chars). Minimum {settings.min_extracted_chars} required. "
                    "The file may be image-based or have extraction issues."
                ),
            )

        result = calculate_ats_score(
            resume_text=resume_text,
            jd_text=jd_text.strip() if jd_text and jd_text.strip() else None,
        )
        logger.info(f"ATS score calculated from file: {result.get('overall_score')}")
        return ATSResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing file upload: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process resume file")


@app.get("/", tags=["Root"])
async def root():
    """Return API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "text_input": "/api/v1/ats-score",
            "file_upload": "/api/v1/ats-score/upload",
        },
        "supported_file_formats": get_supported_extensions(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )