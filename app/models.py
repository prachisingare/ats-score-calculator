"""Pydantic models for request/response validation."""

from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field, field_validator


class ATSRequest(BaseModel):
    """Request model for ATS score calculation."""

    resume_text: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="The candidate's resume text content",
    )

    # ✅ allow short JDs (real world)
    jd_text: Optional[str] = Field(
        default=None,
        min_length=0,
        max_length=30000,
        description="Optional job description text. If not provided, evaluates resume quality standalone.",
    )

    # ✅ normalize whitespace-only jd_text to None (avoids jd_too_short noise)
    @field_validator("jd_text")
    @classmethod
    def _strip_jd_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "resume_text": "John Doe - Software Engineer with 5 years of experience in Python...",
                    "jd_text": "Senior Software Engineer - Python, AWS, Docker...",
                },
                {
                    "resume_text": "Jane Smith - Marketing Manager with 8 years of experience...",
                    "jd_text": None,
                },
            ]
        }
    }


class ScoreBreakdown(BaseModel):
    """Detailed breakdown of ATS score components."""

    keyword_match: float = Field(..., ge=0, le=100, description="Keyword matching score (0-100). 0 in standalone mode.")
    skills_match: float = Field(..., ge=0, le=100, description="Skills matching/density score (0-100).")
    semantic_similarity: float = Field(..., ge=0, le=100, description="Semantic similarity score (0-100). 0 in standalone mode.")
    experience_match: float = Field(..., ge=0, le=100, description="Experience matching score (0-100).")
    education_match: float = Field(..., ge=0, le=100, description="Education matching score (0-100).")
    formatting_score: float = Field(..., ge=0, le=100, description="Resume formatting and structure quality (0-100).")
    completeness_score: float = Field(..., ge=0, le=100, description="Resume completeness score (0-100).")


class ATSResponse(BaseModel):
    """Response model for ATS score calculation."""

    overall_score: float = Field(..., ge=0, le=100, description="Overall ATS compatibility score (0-100).")
    breakdown: ScoreBreakdown = Field(..., description="Detailed score breakdown by category.")

    matched_keywords: List[str] = Field(default_factory=list, description="Keywords found in both resume and JD.")
    missing_keywords: List[str] = Field(default_factory=list, description="JD keywords missing from resume.")
    detected_skills: List[str] = Field(default_factory=list, description="Skills detected in the resume.")

    # ✅ added to match scoring.py output
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills matched between resume and JD (debuggable/optional).",
    )

    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations to improve score.")

    # ✅ constrain allowed values
    mode: Literal["with_jd", "standalone"] = Field(..., description="Evaluation mode: 'with_jd' or 'standalone'.")

    # ✅ keep ONE warnings field only
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings like 'jd_too_short', 'image_based_pdf', 'low_text_extraction'.",
    )

    # ✅ optional production metadata
    score_version: Optional[str] = Field(default=None, description="Scoring logic version (e.g., v1.0).")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model used, if any.")
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Raw cosine similarity (0-1), if computed.",
    )

    # ✅ optional structured recommendations for UI
    recommendations_by_section: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Grouped recommendations, e.g., {'Summary': [...], 'Skills': [...], 'Experience': [...]}",
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service health status.")
    version: str = Field(..., description="API version.")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message.")
    error_code: str = Field(..., description="Error code for programmatic handling.")