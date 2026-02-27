"""Tests for ATS Score Calculator API."""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.scoring import (
    calculate_ats_score,
    extract_keywords,
    extract_years_of_experience,
    calculate_formatting_score,
    calculate_completeness_score,
)
from app.file_parser import (
    extract_text_from_file,
    FileParseError,
    get_supported_extensions,
)

client = TestClient(app)

# Sample test data - Tech domain
SAMPLE_TECH_RESUME = """
John Doe
Senior Software Engineer

Contact: john.doe@email.com | (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe

SUMMARY
Experienced software engineer with 5 years of experience in Python, Django, and cloud technologies.
Strong background in building scalable web applications and microservices architecture.

SKILLS
- Programming: Python, JavaScript, TypeScript, SQL
- Frameworks: Django, Flask, FastAPI, React
- Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
- Databases: PostgreSQL, MongoDB, Redis
- Tools: Git, Jenkins, Jira, Agile/Scrum

EXPERIENCE
Senior Software Engineer | Tech Corp | 2020-Present
- Developed microservices using Python and FastAPI, improving system performance by 40%
- Implemented CI/CD pipelines using Jenkins, reducing deployment time by 60%
- Led team of 4 engineers in delivering critical features

Software Engineer | StartupXYZ | 2018-2020
- Built REST APIs using Django REST Framework
- Deployed applications on AWS, handling 100k+ daily users

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2018
"""

SAMPLE_TECH_JD = """
Senior Software Engineer

We are looking for a Senior Software Engineer to join our team.

Requirements:
- 4+ years of experience in software development
- Strong proficiency in Python and modern frameworks (Django, Flask, or FastAPI)
- Experience with cloud platforms (AWS, GCP, or Azure)
- Knowledge of containerization (Docker, Kubernetes)
- Experience with SQL and NoSQL databases
- Familiarity with CI/CD pipelines
- Bachelor's degree in Computer Science or related field

Nice to have:
- Experience with microservices architecture
- Knowledge of React or Vue.js
- Leadership experience

We offer competitive salary and great work-life balance.
"""

# Sample test data - Healthcare domain
SAMPLE_HEALTHCARE_RESUME = """
Sarah Johnson, RN, BSN
Registered Nurse

Contact: sarah.johnson@email.com | (555) 987-6543

PROFESSIONAL SUMMARY
Compassionate registered nurse with 8 years of experience in critical care and emergency medicine.
Skilled in patient assessment, medication administration, and team collaboration.

LICENSES & CERTIFICATIONS
- Registered Nurse (RN) - State License #12345
- Basic Life Support (BLS) Certified
- Advanced Cardiac Life Support (ACLS) Certified
- Critical Care Registered Nurse (CCRN)

CLINICAL SKILLS
- Patient Assessment & Triage
- IV Therapy & Medication Administration
- Wound Care Management
- Electronic Health Records (Epic, Cerner)
- Ventilator Management
- Patient Education

EXPERIENCE
Critical Care Nurse | City General Hospital | 2019-Present
- Managed care for 4-6 critically ill patients per shift in 20-bed ICU
- Reduced patient fall rate by 35% through improved safety protocols
- Mentored 12 new graduate nurses

Emergency Room Nurse | County Medical Center | 2016-2019
- Triaged and treated 50+ patients daily in high-volume ER
- Collaborated with interdisciplinary team for trauma cases

EDUCATION
Bachelor of Science in Nursing (BSN)
State University School of Nursing | 2016
"""

# Sample test data - Marketing domain
SAMPLE_MARKETING_RESUME = """
Emily Chen
Digital Marketing Manager

Email: emily.chen@email.com | Phone: (555) 456-7890 | Portfolio: emilychen.com

PROFILE
Results-driven digital marketing professional with 6 years of experience in brand management,
content strategy, and campaign optimization. Proven track record of increasing ROI by 150%.

CORE COMPETENCIES
- Digital Marketing Strategy
- SEO/SEM & Google Analytics
- Social Media Marketing
- Content Marketing & Copywriting
- Email Marketing Automation
- Brand Management
- Team Leadership

PROFESSIONAL EXPERIENCE
Digital Marketing Manager | Global Brands Inc. | 2021-Present
- Managed $2M annual digital marketing budget across multiple channels
- Increased organic traffic by 200% through SEO optimization
- Led team of 5 marketing specialists

Marketing Specialist | StartupHub | 2018-2021
- Executed social media campaigns reaching 500k+ followers
- Achieved 45% improvement in email open rates

EDUCATION
Master of Business Administration (MBA), Marketing
Business School University | 2018

Bachelor of Arts in Communications
Liberal Arts College | 2016
"""


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestATSScoreWithJD:
    """Tests for ATS score calculation with job description."""

    def test_tech_resume_with_jd(self):
        response = client.post(
            "/api/v1/ats-score",
            json={"resume_text": SAMPLE_TECH_RESUME, "jd_text": SAMPLE_TECH_JD},
        )
        assert response.status_code == 200
        data = response.json()

        assert "overall_score" in data
        assert 0 <= data["overall_score"] <= 100
        assert data["mode"] == "with_jd"
        assert "breakdown" in data
        assert "matched_keywords" in data
        assert "missing_keywords" in data
        assert "detected_skills" in data
        assert "recommendations" in data

        # ✅ new: warnings field
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

        # Tech resume should score well against tech JD
        assert data["overall_score"] > 50

    def test_validation_error_short_resume(self):
        response = client.post(
            "/api/v1/ats-score",
            json={"resume_text": "Too short", "jd_text": SAMPLE_TECH_JD},
        )
        assert response.status_code == 422

    # ✅ Option B: short JD is allowed, still with_jd, adds warning, semantic similarity forced to 0
    def test_short_jd_still_with_jd_mode_and_warning(self):
        response = client.post(
            "/api/v1/ats-score",
            json={"resume_text": SAMPLE_TECH_RESUME, "jd_text": "Too short"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "with_jd"
        assert "warnings" in data
        assert "jd_too_short" in data["warnings"]
        assert data["breakdown"]["semantic_similarity"] == 0

    def test_missing_resume(self):
        response = client.post("/api/v1/ats-score", json={})
        assert response.status_code == 422


class TestATSScoreStandalone:
    """Tests for standalone ATS score calculation (without JD)."""

    def test_tech_resume_standalone(self):
        response = client.post("/api/v1/ats-score", json={"resume_text": SAMPLE_TECH_RESUME})
        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "standalone"
        assert data["overall_score"] > 0
        assert data["breakdown"]["keyword_match"] == 0
        assert data["breakdown"]["semantic_similarity"] == 0
        assert len(data["matched_keywords"]) == 0
        assert len(data["missing_keywords"]) == 0
        assert len(data["detected_skills"]) > 0

        # warnings should still exist (empty list) if response model includes it
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_healthcare_resume_standalone(self):
        response = client.post("/api/v1/ats-score", json={"resume_text": SAMPLE_HEALTHCARE_RESUME})
        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "standalone"
        assert data["overall_score"] > 50
        assert "detected_skills" in data
        assert "warnings" in data

    def test_marketing_resume_standalone(self):
        response = client.post("/api/v1/ats-score", json={"resume_text": SAMPLE_MARKETING_RESUME})
        assert response.status_code == 200
        data = response.json()

        assert data["mode"] == "standalone"
        assert data["overall_score"] > 50
        assert "warnings" in data

    def test_null_jd_is_standalone(self):
        response = client.post("/api/v1/ats-score", json={"resume_text": SAMPLE_TECH_RESUME, "jd_text": None})
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "standalone"
        assert "warnings" in data


class TestScoringModule:
    """Tests for the scoring module functions."""

    def test_extract_keywords(self):
        text = "Python developer with Django experience"
        keywords = extract_keywords(text)
        assert "python" in keywords
        assert "developer" in keywords
        assert "django" in keywords

    def test_extract_years_of_experience(self):
        text = "5 years of experience in software development"
        years = extract_years_of_experience(text)
        assert 5 in years

        text2 = "Over 10 years in healthcare"
        years2 = extract_years_of_experience(text2)
        assert 10 in years2

    def test_formatting_score(self):
        score = calculate_formatting_score(SAMPLE_TECH_RESUME)
        assert 0 <= score <= 100
        assert score > 50

    def test_completeness_score(self):
        score = calculate_completeness_score(SAMPLE_TECH_RESUME)
        assert 0 <= score <= 100
        assert score > 50

    def test_calculate_ats_score_with_jd(self):
        result = calculate_ats_score(SAMPLE_TECH_RESUME, SAMPLE_TECH_JD)

        assert result["mode"] == "with_jd"
        assert "overall_score" in result
        assert "breakdown" in result
        assert "matched_keywords" in result
        assert "missing_keywords" in result
        assert "detected_skills" in result
        assert "recommendations" in result

        # ✅ new
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

        assert result["overall_score"] > 50

    def test_calculate_ats_score_standalone(self):
        result = calculate_ats_score(SAMPLE_TECH_RESUME)

        assert result["mode"] == "standalone"
        assert result["overall_score"] > 0
        assert result["breakdown"]["keyword_match"] == 0
        assert result["breakdown"]["semantic_similarity"] == 0
        assert "warnings" in result

    def test_different_domains_score_reasonably(self):
        tech_result = calculate_ats_score(SAMPLE_TECH_RESUME)
        healthcare_result = calculate_ats_score(SAMPLE_HEALTHCARE_RESUME)
        marketing_result = calculate_ats_score(SAMPLE_MARKETING_RESUME)

        assert tech_result["overall_score"] > 40
        assert healthcare_result["overall_score"] > 40
        assert marketing_result["overall_score"] > 40

    # ✅ recommended: determinism
    def test_score_deterministic(self):
        r1 = calculate_ats_score(SAMPLE_TECH_RESUME, SAMPLE_TECH_JD)
        r2 = calculate_ats_score(SAMPLE_TECH_RESUME, SAMPLE_TECH_JD)
        assert r1["overall_score"] == r2["overall_score"]


class TestBreakdownFields:
    """Tests for score breakdown fields."""

    def test_all_breakdown_fields_present(self):
        response = client.post("/api/v1/ats-score", json={"resume_text": SAMPLE_TECH_RESUME})
        data = response.json()
        breakdown = data["breakdown"]

        required_fields = [
            "keyword_match",
            "skills_match",
            "semantic_similarity",
            "experience_match",
            "education_match",
            "formatting_score",
            "completeness_score",
        ]

        for field in required_fields:
            assert field in breakdown
            assert 0 <= breakdown[field] <= 100


class TestFileUpload:
    """Tests for file upload endpoint."""

    def test_upload_txt_file_standalone(self):
        file_content = SAMPLE_TECH_RESUME.encode("utf-8")

        response = client.post(
            "/api/v1/ats-score/upload",
            files={"resume_file": ("resume.txt", io.BytesIO(file_content), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "standalone"
        assert data["overall_score"] > 0
        assert "warnings" in data

    def test_upload_txt_file_with_jd(self):
        file_content = SAMPLE_TECH_RESUME.encode("utf-8")

        response = client.post(
            "/api/v1/ats-score/upload",
            files={"resume_file": ("resume.txt", io.BytesIO(file_content), "text/plain")},
            data={"jd_text": SAMPLE_TECH_JD},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "with_jd"
        assert data["overall_score"] > 50
        assert "warnings" in data

    def test_upload_unsupported_format(self):
        file_content = b"Some content"

        response = client.post(
            "/api/v1/ats-score/upload",
            files={"resume_file": ("resume.xyz", io.BytesIO(file_content), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_upload_empty_file(self):
        response = client.post(
            "/api/v1/ats-score/upload",
            files={"resume_file": ("resume.txt", io.BytesIO(b""), "text/plain")},
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_file_with_short_content(self):
        file_content = b"Too short"

        response = client.post(
            "/api/v1/ats-score/upload",
            files={"resume_file": ("resume.txt", io.BytesIO(file_content), "text/plain")},
        )

        assert response.status_code == 422
        assert "too short" in response.json()["detail"].lower()


class TestFileParser:
    """Tests for file parser module."""

    def test_get_supported_extensions(self):
        extensions = get_supported_extensions()
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".txt" in extensions

    def test_extract_text_from_txt(self):
        content = SAMPLE_TECH_RESUME.encode("utf-8")
        file_obj = io.BytesIO(content)

        text = extract_text_from_file(file_obj, "resume.txt")
        assert "John Doe" in text
        assert "Software Engineer" in text

    def test_unsupported_extension_raises_error(self):
        file_obj = io.BytesIO(b"content")

        with pytest.raises(FileParseError) as exc_info:
            extract_text_from_file(file_obj, "resume.xyz")

        assert "Unsupported file type" in str(exc_info.value)

    def test_legacy_doc_format_raises_error(self):
        file_obj = io.BytesIO(b"content")

        with pytest.raises(FileParseError) as exc_info:
            extract_text_from_file(file_obj, "resume.doc")

        assert ".doc format is not supported" in str(exc_info.value)


class TestRootEndpointUpdated:
    """Tests for updated root endpoint."""

    def test_root_contains_endpoints_info(self):
        response = client.get("/")
        data = response.json()

        assert "endpoints" in data
        assert "text_input" in data["endpoints"]
        assert "file_upload" in data["endpoints"]
        assert "supported_file_formats" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])