"""
ATS Scoring Algorithm (Domain-Agnostic, JD-Driven) — Production Ready (FAST + SAFE)

v2.8 fixes:
✅ Skills noise filtering: months, urls/emails, placeholders, generic resume words, address words, numbers/years.
✅ JD keyword cleaning: removes template/junk phrases, disables 3-grams, filters long fragments, removes verb-y junk.
✅ Single-word skills: keeps real single-word skills (venipuncture) while dropping junk (stellar/technique/puncture).
✅ FIX: _chunk_sentences uses _SENT_SPLIT (bugfix).
"""

from __future__ import annotations

import re
import logging
from typing import TypedDict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from app.config import settings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------
# Optional: sentence-transformers
# ---------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    logger.warning(f"sentence_transformers import failed: {e}")

try:
    import torch  # type: ignore
    torch.set_num_threads(2)
except Exception:
    pass

EMBED_MODEL: Optional["SentenceTransformer"] = None
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def init_embedding_model() -> None:
    global EMBED_MODEL
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        EMBED_MODEL = None
        return
    if EMBED_MODEL is not None:
        return
    try:
        EMBED_MODEL = SentenceTransformer(settings.embedding_model_name)
        logger.info(f"✅ SentenceTransformer loaded: {settings.embedding_model_name}")
    except Exception as e:
        EMBED_MODEL = None
        logger.warning(f"Failed to load embedding model: {e}")


def get_embedding_model():
    return EMBED_MODEL


# ---------------------------
# Types
# ---------------------------
class ScoreBreakdown(TypedDict):
    keyword_match: float
    skills_match: float
    semantic_similarity: float
    experience_match: float
    education_match: float
    formatting_score: float
    completeness_score: float


class ATSResult(TypedDict, total=False):
    overall_score: float
    breakdown: ScoreBreakdown
    matched_keywords: list[str]
    missing_keywords: list[str]
    detected_skills: list[str]
    matched_skills: list[str]
    recommendations: list[str]
    mode: str
    warnings: list[str]
    score_version: str
    embedding_model: Optional[str]
    similarity_score: Optional[float]
    recommendations_by_section: Optional[dict]


# ---------------------------
# Constants
# ---------------------------
STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","as","is","was","are","were",
    "been","be","have","has","had","do","does","did","will","would","could","should","may","might","must","shall","can",
    "need","we","you","i","he","she","it","they","them","their","our","your","its","this","that","these","those",
    "what","which","who","whom","where","when","why","how","all","each","every","both","few","more","most","other",
    "some","such","no","nor","not","only","own","same","so","than","too","very","just","about","after","before",
    "during","through","between","into","also","etc"
}

FLUFF_PHRASES = {
    "ideal candidate", "ideal", "candidate", "comfortable", "comfortable working",
    "fast paced", "self starter", "detail oriented", "team player",
    "strong communication", "good communication", "work independently",
    "responsible for", "ability to", "must have", "nice to have",
    "highly motivated", "results driven", "results-oriented", "hard working",
    "fast learner", "passionate", "seeking an opportunity", "looking for",
    "career objective", "proven track record", "excellent communication",
    "strong interpersonal", "ability to work",
}

EDUCATION_KEYWORDS = {
    "bachelor","master","phd","doctorate","mba","degree","bs","ms","ba","ma","bsc","msc",
    "diploma","certificate","certification","certified","university","college","institute",
    "school","academy","graduate","undergraduate","postgraduate","associate","honors","distinction"
}

ACTION_VERBS = {
    "achieved","accomplished","administered","analyzed","built","coordinated","created","delivered",
    "developed","directed","established","executed","generated","implemented","improved","increased",
    "initiated","launched","led","managed","optimized","organized","planned","produced","reduced",
    "resolved","streamlined","supervised","trained","designed","owned","drove","automated",
    "negotiated","presented","reported","communicated","collaborated","supported","monitored"
}

MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,
    "jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}

MONTH_WORDS = set(MONTHS.keys())

GENERIC_RESUME_WORDS = {
    "summary","skills","responsibility","responsibilities","profile","objective",
    "experience","education","project","projects","certification","certifications",
    "award","awards","achievement","achievements","contact","references"
}

ADDRESS_WORDS = {
    "street","st","road","rd","ave","avenue","blvd","boulevard","lane","ln","zip",
    "city","state","country","address"
}

SKILL_NOISE_WORDS = MONTH_WORDS | GENERIC_RESUME_WORDS | ADDRESS_WORDS

COMMON_NON_SKILLS = {
    "ability","accomplishment","according","bellow","colleague","completed","course",
    "established","ethics","excellent","familiar","health","info","introduction","key",
    "law","medical","national","organization","part","performing","place","present",
    "privacy","processing","professional","program","recognized","responsible","responsibility",
    "provide","environment","follow","handling","maintain","ensure","control","properly",
}

PLACEHOLDER_PATTERNS = [
    r"\bexample\.com\b",
    r"\btest\.com\b",
    r"\bmailinator\.com\b",
]

JD_FLUFF_LINE_PATTERNS = [
    r"^\s*(about\s+us|company|overview|who\s+we\s+are)\s*:?\s*$",
    r"^\s*(job\s+title|title|location|employment\s+type|salary|compensation)\s*:?\s*.*$",
    r"^\s*(responsibilities|requirements|preferred|nice\s+to\s+have|must\s+have)\s*:?\s*$",
    r"^\s*(equal\s+opportunity|eeo|diversity|inclusion|benefits)\s*:?\s*$",
]

JD_BAD_EDGE_WORDS = {
    "job","title","preferred","required","must","have","nice","to",
    "responsibilities","requirements"
}

JD_TEMPLATE_PHRASES = {
    "job title",
    "job title certified",
    "employment type",
    "equal opportunity",
    "about us",
    "who we are",
    "benefits",
    "salary",
    "location",
}

# Remove verb-y TF-IDF junk from JD keywords
JD_NOISE_EDGE_TOKENS = {
    "provide","provided","providing",
    "perform","performed","performing",
    "ensure","ensuring",
    "maintain","maintaining",
    "follow","following",
    "seeking","seek",
    "record","records","recording",
    "relevant",
    "setting","settings",
    "process","processing",
    "properly",
    "environment",
}

# Drop generic single-word junk from "skills"
GENERIC_SINGLE_WORD_SKILLS = {
    "stellar", "technique", "techniques", "puncture", "practice", "practices"
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s\+\#\.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_fluff(phrase: str) -> str:
    p = (phrase or "").strip().lower()
    p = re.sub(r"\s+", " ", p)
    p = p.strip(" .,:;|-_/\\")
    return p


def _looks_like_url_or_email(s: str) -> bool:
    s = (s or "").strip().lower()
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", s):
        return True
    if "http://" in s or "https://" in s or "www." in s:
        return True
    if any(re.search(p, s) for p in PLACEHOLDER_PATTERNS):
        return True
    return False


def is_noise_skill(p: str) -> bool:
    p = strip_fluff(p)
    if not p:
        return True
    if _looks_like_url_or_email(p):
        return True
    if re.fullmatch(r"(19|20)\d{2}", p) or re.fullmatch(r"\d{1,4}", p):
        return True
    if p in SKILL_NOISE_WORDS:
        return True
    if p in COMMON_NON_SKILLS:
        return True
    if " " not in p and p in GENERIC_SINGLE_WORD_SKILLS:
        return True
    if len(p) <= 2:
        return True
    return False


def is_good_phrase(p: str) -> bool:
    if not p or len(p) < 3:
        return False
    if p in STOP_WORDS or p in FLUFF_PHRASES:
        return False
    words = p.split()
    if not words:
        return False
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return False
    stop_ratio = sum(1 for w in words if w in STOP_WORDS) / max(1, len(words))
    if stop_ratio > 0.4:
        return False
    if all(w.isdigit() for w in words):
        return False
    if len(words) == 3 and sum(len(w) <= 3 for w in words) >= 2:
        return False
    return True


def is_jd_fragment(p: str) -> bool:
    p = strip_fluff(p)
    if not p:
        return True
    if p in JD_TEMPLATE_PHRASES:
        return True
    toks = p.split()
    if len(toks) >= 5:
        return True
    if toks and (toks[0] in JD_BAD_EDGE_WORDS or toks[-1] in JD_BAD_EDGE_WORDS):
        return True
    stop_ratio = sum(1 for w in toks if w in STOP_WORDS) / max(1, len(toks))
    if stop_ratio > 0.34:
        return True
    if "job title" in p:
        return True
    return False


def extract_tech_tokens(text: str) -> set[str]:
    t = preprocess_text(text)
    toks = set(re.findall(r"\b[a-z][a-z0-9\+\#\.\-/]{1,}\b", t))
    out: set[str] = set()
    for x in toks:
        x = strip_fluff(x)
        if not x or x in STOP_WORDS or x.isdigit():
            continue
        if is_noise_skill(x):
            continue
        if "/" in x and len(x) <= 40:
            parts = [strip_fluff(p) for p in x.split("/") if p]
            for p in parts:
                if is_good_phrase(p) and not is_noise_skill(p):
                    out.add(p)
            continue
        if is_good_phrase(x) and not is_noise_skill(x):
            out.add(x)
    return out


def normalize_skill(s: str) -> str:
    s = strip_fluff(s)
    s = s.replace("powerbi", "power bi")
    s = s.replace("scikit learn", "scikit-learn")
    s = s.replace("ci cd", "ci/cd")
    s = re.sub(r"\s+", " ", s)
    return s


_CANON_MAP = {
    "cnns": "cnn",
    "cnn": "cnn",
    "transformers": "transformer",
    "embeddings": "embedding",
    "deploy": "deployment",
    "deployed": "deployment",
    "deploying": "deployment",
    "ci cd": "ci/cd",
    "git hub": "github",
    "scikit learn": "scikit-learn",
    "powerbi": "power bi",
}

NO_NORMALIZE_EXACT = {"faiss", "pandas", "keras", "xgboost", "lightgbm", "scikit-learn", "langchain", "github", "mlops"}


def _safe_depluralize(p: str) -> str:
    if p in NO_NORMALIZE_EXACT:
        return p
    if p.endswith("ies") and len(p) > 5:
        return p[:-3] + "y"
    if p.endswith("s") and len(p) > 6 and not p.endswith("ss"):
        return p[:-1]
    return p


def canon_phrase(p: str) -> str:
    p = normalize_skill(p)
    p = p.replace("-", " ")
    p = re.sub(r"\s+", " ", p).strip()
    p = _CANON_MAP.get(p, p)
    if p in NO_NORMALIZE_EXACT:
        return p
    p2 = _safe_depluralize(p)
    return _CANON_MAP.get(p2, p2)


def phrase_tokens(p: str) -> set[str]:
    return {w for w in canon_phrase(p).split() if w and w not in STOP_WORDS}


def jd_top_phrases_tfidf(jd_text: str, top_k: int = 80) -> set[str]:
    jd = preprocess_text(jd_text)
    if len(jd) < 60:
        return set()

    docs = [preprocess_text(x) for x in (jd_text or "").splitlines()]
    docs = [d for d in docs if len(d.split()) >= 4]
    if len(docs) < 3:
        docs = [s.strip() for s in _SENT_SPLIT.split(jd) if len(s.split()) >= 4]
    if not docs:
        docs = [jd]

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=6000,
        min_df=1,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    feats = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1

    idx = scores.argsort()[::-1][:top_k]
    out = set()
    for i in idx:
        p = strip_fluff(feats[i])
        if not is_good_phrase(p):
            continue
        if is_jd_fragment(p):
            continue
        out.add(p)
    return out


def extract_resume_keywords(resume_text: str) -> set[str]:
    t = preprocess_text(resume_text)
    words = [w for w in t.split() if w and w not in STOP_WORDS and len(w) >= 3]
    unigrams = set(words)

    bigrams = set()
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if is_good_phrase(bg):
            bigrams.add(bg)

    raw = unigrams | bigrams | extract_tech_tokens(t)
    out = set()
    for x in raw:
        x = strip_fluff(x)
        if is_good_phrase(x):
            cx = canon_phrase(x)
            if not is_noise_skill(cx):
                out.add(cx)
    return out


def extract_jd_keywords(jd_text: str) -> set[str]:
    top = jd_top_phrases_tfidf(jd_text, top_k=90)
    tech = extract_tech_tokens(jd_text)
    raw = top | tech

    generic = {
        "experience","skills","knowledge","ability","work","team","role","responsibilities",
        "requirements","preferred","qualification","qualifications","about","company",
        "benefits","salary","location","job","position","overview","summary"
    }

    out = set()
    for x in raw:
        x = strip_fluff(x)
        if not is_good_phrase(x):
            continue
        if x in generic:
            continue

        cx = canon_phrase(x)
        if is_noise_skill(cx):
            continue
        if is_jd_fragment(cx):
            continue

        toks = cx.split()
        if toks and (toks[0] in JD_NOISE_EDGE_TOKENS or toks[-1] in JD_NOISE_EDGE_TOKENS):
            continue
        if sum(1 for t in toks if t in JD_NOISE_EDGE_TOKENS) >= 2:
            continue

        out.add(cx)

    if len(out) > 160:
        out = set(sorted(out)[:160])

    return out


def calculate_keyword_match(resume_keywords: set[str], jd_keywords: set[str]) -> Tuple[float, list[str], list[str]]:
    if not jd_keywords:
        return 100.0, [], []

    matched = set()
    resume_token_bag = set()
    for rk in resume_keywords:
        resume_token_bag |= phrase_tokens(rk)

    for jk in jd_keywords:
        if jk in resume_keywords:
            matched.add(jk)
            continue
        jt = phrase_tokens(jk)
        if jt and jt.issubset(resume_token_bag):
            matched.add(jk)

    missing = jd_keywords - matched
    score = (len(matched) / len(jd_keywords)) * 100 if jd_keywords else 100.0
    return float(score), sorted(matched)[:50], sorted(missing)[:50]


def extract_skill_candidates_from_lists(text: str) -> set[str]:
    t = preprocess_text(text)
    candidates: set[str] = set()
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if (s.count(",") >= 2) or ("|" in s) or (s.count("/") >= 2):
            parts = re.split(r"[,\|•\u2022;/]+", s)
            for p in parts:
                p = strip_fluff(p)
                if is_good_phrase(p) and len(p.split()) <= 5:
                    candidates.add(p)
    return candidates


def extract_skill_candidates_from_patterns(text: str) -> set[str]:
    t = preprocess_text(text)
    candidates: set[str] = set()
    patterns = [
        r"(?:experience with|proficient in|knowledge of|familiarity with|expertise in|hands on|hands-on)\s+([^.\n]{5,140})",
        r"(?:tools|technologies|tech stack|stack|platforms)\s*:\s*([^.\n]{5,160})",
        r"(?:requirements|must have|skills)\s*:\s*([^.\n]{5,180})",
    ]
    for pat in patterns:
        for m in re.findall(pat, t):
            chunk = m.strip()
            parts = re.split(r"[,\|•\u2022;/]+", chunk)
            for p in parts:
                p = strip_fluff(p)
                if is_good_phrase(p) and len(p.split()) <= 6:
                    candidates.add(p)
    return candidates


def _filter_single_word_skills(skills: set[str], full_text: str) -> set[str]:
    t = preprocess_text(full_text)
    filtered = set()
    for s in skills:
        if " " in s:
            filtered.add(s)
            continue
        if any(ch in s for ch in ["/", "-", ".", "+", "#"]):
            filtered.add(s)
            continue
        if len(s) >= 4 and not is_noise_skill(s):
            filtered.add(s)
            continue
    return filtered


def extract_skills_jd(jd_text: str) -> set[str]:
    t = preprocess_text(jd_text)
    cands = set()
    cands |= extract_skill_candidates_from_lists(t)
    cands |= extract_skill_candidates_from_patterns(t)
    cands |= extract_tech_tokens(t)

    skills = set()
    for c in cands:
        c2 = canon_phrase(c)
        if is_noise_skill(c2):
            continue
        if is_good_phrase(c2) and len(c2) <= 50:
            skills.add(c2)

    return _filter_single_word_skills(skills, jd_text)


def extract_skills_resume(resume_text: str) -> set[str]:
    t = preprocess_text(resume_text)
    cands = set()
    cands |= extract_skill_candidates_from_lists(t)
    cands |= extract_skill_candidates_from_patterns(t)
    cands |= extract_tech_tokens(t)

    skills = set()
    for c in cands:
        c2 = canon_phrase(c)
        if is_noise_skill(c2):
            continue
        if is_good_phrase(c2) and len(c2) <= 50:
            skills.add(c2)

    return _filter_single_word_skills(skills, resume_text)


def calculate_skills_match(resume_text: str, jd_text: Optional[str] = None) -> Tuple[float, list[str], list[str]]:
    resume_skills = extract_skills_resume(resume_text)

    if jd_text:
        jd_skills = extract_skills_jd(jd_text)
        if not jd_skills:
            return 100.0, sorted(resume_skills)[:40], []

        matched = set()
        resume_token_bag = set()
        for rs in resume_skills:
            resume_token_bag |= phrase_tokens(rs)

        for js in jd_skills:
            if js in resume_skills:
                matched.add(js)
                continue
            jt = phrase_tokens(js)
            if jt and jt.issubset(resume_token_bag):
                matched.add(js)

        score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 100.0
        return float(score), sorted(resume_skills)[:50], sorted(matched)[:50]

    density = min(len(resume_skills) / 18, 1.0) * 100
    return float(density), sorted(resume_skills)[:50], []


def extract_years_explicit(text: str) -> list[float]:
    t = preprocess_text(text)
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)\b",
        r"\b(?:experience|exp)\s*(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b",
        r"\bover\s+(\d+(?:\.\d+)?)\s+(?:years?|yrs?)\b",
    ]
    out: list[float] = []
    for p in patterns:
        for y in re.findall(p, t):
            try:
                v = float(y)
                if 0 < v < 60:
                    out.append(v)
            except Exception:
                pass
    return out


def _parse_month_year(s: str) -> Optional[Tuple[int, int]]:
    s = s.strip().lower()
    m1 = re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+((19|20)\d{2})$", s)
    if m1:
        return (int(m1.group(2)), MONTHS[m1.group(1)])

    m2 = re.match(r"^(\d{1,2})[\/\-](\d{4})$", s)
    if m2:
        mm = int(m2.group(1))
        yy = int(m2.group(2))
        if 1 <= mm <= 12:
            return (yy, mm)

    m3 = re.match(r"^((19|20)\d{2})$", s)
    if m3:
        return (int(m3.group(1)), 1)

    return None


def extract_years_from_date_ranges(text: str, present_proxy: Tuple[int, int] = (2026, 2)) -> float:
    t = preprocess_text(text).replace("–", "-").replace("—", "-")
    pattern = re.compile(
        r"\b([a-z]{3,9}\s+(?:19|20)\d{2}|\d{1,2}[\/\-](?:19|20)\d{2}|(?:19|20)\d{2})\s*-\s*"
        r"(present|current|now|[a-z]{3,9}\s+(?:19|20)\d{2}|\d{1,2}[\/\-](?:19|20)\d{2}|(?:19|20)\d{2})\b"
    )

    spans_months: list[int] = []
    for a, b in pattern.findall(t):
        start = _parse_month_year(a)
        end = present_proxy if b in {"present", "current", "now"} else _parse_month_year(b)
        if not start or not end:
            continue
        sy, sm = start
        ey, em = end
        if (ey, em) < (sy, sm):
            continue
        months = (ey - sy) * 12 + (em - sm) + 1
        if 1 <= months <= 12 * 40:
            spans_months.append(months)

    if not spans_months:
        return 0.0
    return round(max(spans_months) / 12.0, 2)


def extract_years_of_experience(text: str) -> list[float]:
    years = extract_years_explicit(text)
    span = extract_years_from_date_ranges(text)
    if span > 0:
        years.append(span)
    return years


def calculate_experience_score(resume_text: str, jd_text: Optional[str] = None) -> float:
    resume_years = extract_years_of_experience(resume_text)
    cand = max(resume_years) if resume_years else 0.0

    rlow = preprocess_text(resume_text)
    has_intern_or_proj = ("intern" in rlow) or ("internship" in rlow) or ("project" in rlow)

    if jd_text:
        jd_years = extract_years_of_experience(jd_text)
        if not jd_years:
            return 85.0 if (cand > 0 or has_intern_or_proj) else 70.0

        required = max(jd_years)
        if cand <= 0:
            return 50.0

        ratio = cand / required if required > 0 else 1.0
        if ratio >= 1.0:
            return 100.0
        if ratio >= 0.85:
            return 90.0
        if ratio >= 0.70:
            return 80.0
        if ratio >= 0.55:
            return 70.0
        if ratio >= 0.40:
            return 60.0
        return 45.0

    if cand >= 5:
        return 100.0
    if cand >= 3:
        return 85.0
    if cand >= 1:
        return 70.0
    return 65.0 if has_intern_or_proj else (60.0 if "experience" in rlow else 50.0)


def calculate_education_score(resume_text: str, jd_text: Optional[str] = None) -> float:
    r = preprocess_text(resume_text)
    resume_edu = {kw for kw in EDUCATION_KEYWORDS if kw in r}

    if jd_text:
        j = preprocess_text(jd_text)
        jd_edu = {kw for kw in EDUCATION_KEYWORDS if kw in j}
        if not jd_edu:
            return 100.0 if resume_edu else 80.0
        matched = resume_edu & jd_edu
        return float((len(matched) / len(jd_edu)) * 100) if jd_edu else 100.0

    if not resume_edu:
        return 50.0
    high = {"phd","doctorate","master","mba","ms","msc","ma"}
    bachelor = {"bachelor","bs","bsc","ba","degree"}
    if resume_edu & high:
        return 100.0
    if resume_edu & bachelor:
        return 85.0
    if "certificate" in resume_edu or "certification" in resume_edu:
        return 70.0
    return 60.0


def extract_contact_info(text: str) -> dict:
    t = text or ""
    tl = t.lower()
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", t))
    has_phone = bool(re.search(r"[\+]?[\d\s\-\(\)]{10,}", t))
    has_linkedin = "linkedin" in tl
    has_location = any(x in tl for x in [" city", " state", " country", "address", ","])
    return {"email": has_email, "phone": has_phone, "linkedin": has_linkedin, "location": has_location}


def calculate_formatting_score(resume_text: str) -> float:
    t = preprocess_text(resume_text)
    score = 0.0

    section_terms = [
        "experience","work experience","professional experience",
        "education","skills","projects","project",
        "summary","professional summary","profile","objective",
        "certifications","certification","awards","achievements",
    ]
    section_hits = sum(1 for s in section_terms if s in t)
    score += min(section_hits / 6, 1.0) * 35

    contact = extract_contact_info(resume_text)
    score += (sum(contact.values()) / 4) * 20

    wc = len((resume_text or "").split())
    if 200 <= wc <= 1500:
        score += 20
    elif 100 <= wc < 200 or 1500 < wc <= 2000:
        score += 10

    action_count = sum(1 for v in ACTION_VERBS if v in t)
    score += min(action_count / 8, 1.0) * 15

    numbers = re.findall(r"\d+%|\$\d+|\d+\s*(?:million|billion|thousand|lakh|crore|k\b)", t)
    score += min(len(numbers) / 3, 1.0) * 10

    return float(min(score, 100.0))


def calculate_completeness_score(resume_text: str) -> float:
    t = preprocess_text(resume_text)
    score = 0.0

    for section in ["experience","education","skills"]:
        if section in t:
            score += 20

    contact = extract_contact_info(resume_text)
    if contact["email"]:
        score += 15
    if contact["phone"]:
        score += 10

    if re.search(r"\b(19|20)\d{2}\b", resume_text or ""):
        score += 10
    if re.search(r"\d+", resume_text or ""):
        score += 5

    return float(min(score, 100.0))


def _preprocess_for_similarity(text: str) -> str:
    t = preprocess_text(text)
    words = [w for w in t.split() if len(w) >= 2 and w not in STOP_WORDS]
    return " ".join(words)


def _remove_jd_fluff_lines(jd_text: str) -> str:
    lines = (jd_text or "").splitlines()
    cleaned: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(re.match(pat, low) for pat in JD_FLUFF_LINE_PATTERNS):
            continue
        cleaned.append(s)
    return "\n".join(cleaned)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _chunk_sentences(text: str, target_words: int = 250, max_chunks: int = 10) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # ✅ FIX: use _SENT_SPLIT (not __SENT_SPLIT)
    sents = _SENT_SPLIT.split(text)

    chunks: List[str] = []
    buf: List[str] = []
    wc = 0

    for s in sents:
        s = s.strip()
        if not s:
            continue
        w = len(s.split())

        if w > target_words:
            if buf:
                chunks.append(" ".join(buf).strip())
                buf, wc = [], 0
                if len(chunks) >= max_chunks:
                    break
            chunks.append(" ".join(s.split()[:target_words]))
            if len(chunks) >= max_chunks:
                break
            continue

        if wc + w > target_words and buf:
            chunks.append(" ".join(buf).strip())
            buf, wc = [s], w
            if len(chunks) >= max_chunks:
                break
        else:
            buf.append(s)
            wc += w

    if len(chunks) < max_chunks and buf:
        chunks.append(" ".join(buf).strip())

    return chunks[:max_chunks]


def _embed_and_score(model, r_chunks: List[str], j_chunks: List[str], batch_size: int) -> float:
    r_emb = model.encode(r_chunks, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    j_emb = model.encode(j_chunks, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    sims = util.cos_sim(r_emb, j_emb)
    max_sim = float(sims.max().item())
    return float(max(0.0, min(100.0, max_sim * 100)))


def calculate_semantic_similarity(
    resume_text: str,
    jd_text: str,
    use_embeddings: bool = True,
    max_resume_chars: int = 12000,
    max_jd_chars: int = 8000,
    resume_max_chunks: int = 10,
    jd_max_chunks: int = 6,
    target_words_per_chunk: int = 250,
    batch_size: int = 8,
    max_seconds: float = 6.0,
) -> float:
    if not (resume_text or "").strip() or not (jd_text or "").strip():
        return 0.0

    jd_text2 = _remove_jd_fluff_lines(jd_text)

    r = _preprocess_for_similarity(_truncate_text(resume_text, max_resume_chars))
    j = _preprocess_for_similarity(_truncate_text(jd_text2, max_jd_chars))

    if len(r) < 50 or len(j) < 50:
        return 0.0

    if use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
        model = get_embedding_model()
        if model is not None:
            try:
                r_chunks = _chunk_sentences(r, target_words=target_words_per_chunk, max_chunks=resume_max_chunks)
                j_chunks = _chunk_sentences(j, target_words=target_words_per_chunk, max_chunks=jd_max_chunks)
                if not r_chunks or not j_chunks:
                    return 0.0

                fut = _EMBED_EXECUTOR.submit(_embed_and_score, model, r_chunks, j_chunks, batch_size)
                return float(fut.result(timeout=max_seconds))
            except FuturesTimeout:
                logger.warning("Embedding similarity timed out; using TF-IDF fallback")
            except Exception as e:
                logger.warning(f"Embedding similarity failed: {e}; using TF-IDF fallback")

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    try:
        X = vec.fit_transform([r, j])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(max(0.0, min(100.0, sim * 100)))
    except Exception:
        return 0.0


def generate_recommendations(breakdown: ScoreBreakdown, missing_keywords: list[str], mode: str) -> list[str]:
    recs: list[str] = []

    def add(msg: str):
        if msg and msg not in recs:
            recs.append(msg)

    if mode == "with_jd":
        if missing_keywords:
            add("Add missing JD keywords naturally in Summary/Skills/Experience.")
            add(f"Prioritize these missing keywords: {', '.join(missing_keywords[:8])}")
        if breakdown["keyword_match"] < 80:
            add("Mirror key JD phrases in Experience bullets (tools, responsibilities, domain terms).")
        if breakdown["skills_match"] < 75:
            add("Add required skills/tools from JD under Skills and in 2–3 bullets/projects.")
        if breakdown["experience_match"] < 80:
            add("Make timeline explicit: Company, Role, Month YYYY–Month YYYY (and any relevant years).")
        if breakdown["education_match"] < 80:
            add("Include required degree/certification keywords in Education/Certifications.")

    if breakdown["formatting_score"] < 85:
        add("Use clear headers: SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS.")
        add("Use bullets: Action verb + task + tools + measurable result.")

    if breakdown["completeness_score"] < 90:
        add("Ensure contact info includes Email, Phone, LinkedIn, and Location.")

    add("Add measurable impact in 3–6 bullets (%, time saved, revenue, cost, SLA, quality).")
    add("Add 2–3 projects/initiatives with problem + approach + tools + result + link (portfolio/GitHub).")
    add("Place top JD keywords in: Summary (2–3), Skills (8–12), and Experience bullets (3–6).")
    add("Avoid tables/text boxes/multi-column layout; use a single-column ATS-friendly format.")

    return recs[:12]


def calculate_ats_score(resume_text: str, jd_text: Optional[str] = None) -> ATSResult:
    mode = "with_jd" if jd_text and str(jd_text).strip() else "standalone"
    warnings: list[str] = []

    formatting_score = calculate_formatting_score(resume_text)
    completeness_score = calculate_completeness_score(resume_text)
    education_score = calculate_education_score(resume_text, jd_text if mode == "with_jd" else None)

    skills_score, detected_skills, matched_skills = calculate_skills_match(resume_text, jd_text if mode == "with_jd" else None)
    experience_score = calculate_experience_score(resume_text, jd_text if mode == "with_jd" else None)

    score_version = "v2.8-domain-agnostic-cleaner-keywords-skills"
    embedding_model: Optional[str] = None
    similarity_score: Optional[float] = None

    if mode == "with_jd":
        jd_clean = _preprocess_for_similarity(jd_text or "")
        if len(jd_clean) < 50:
            warnings.append("jd_too_short")

        resume_kw = extract_resume_keywords(resume_text)
        jd_kw = extract_jd_keywords(jd_text or "")
        keyword_score, matched_keywords, missing_keywords = calculate_keyword_match(resume_kw, jd_kw)

        semantic_score = 0.0 if "jd_too_short" in warnings else calculate_semantic_similarity(
            resume_text=resume_text,
            jd_text=jd_text or "",
            use_embeddings=settings.use_embeddings,
            max_resume_chars=settings.max_resume_chars,
            max_jd_chars=settings.max_jd_chars,
            resume_max_chunks=settings.resume_max_chunks,
            jd_max_chunks=settings.jd_max_chunks,
            target_words_per_chunk=settings.target_words_per_chunk,
            batch_size=settings.embedding_batch_size,
            max_seconds=settings.embedding_timeout_seconds,
        )
        similarity_score = 0.0 if "jd_too_short" in warnings else round(semantic_score / 100.0, 4)

        if SENTENCE_TRANSFORMERS_AVAILABLE and "jd_too_short" not in warnings and get_embedding_model() is not None:
            embedding_model = settings.embedding_model_name

        weights = {
            "keyword_match": 0.25,
            "skills_match": 0.30,
            "semantic_similarity": 0.10,
            "experience_match": 0.15,
            "education_match": 0.08,
            "formatting_score": 0.08,
            "completeness_score": 0.04,
        }
    else:
        keyword_score = 0.0
        matched_keywords = []
        missing_keywords = []
        semantic_score = 0.0
        weights = {
            "keyword_match": 0.0,
            "skills_match": 0.35,
            "semantic_similarity": 0.0,
            "experience_match": 0.25,
            "education_match": 0.15,
            "formatting_score": 0.15,
            "completeness_score": 0.10,
        }

    overall_score = (
        keyword_score * weights["keyword_match"]
        + skills_score * weights["skills_match"]
        + semantic_score * weights["semantic_similarity"]
        + experience_score * weights["experience_match"]
        + education_score * weights["education_match"]
        + formatting_score * weights["formatting_score"]
        + completeness_score * weights["completeness_score"]
    )

    breakdown: ScoreBreakdown = {
        "keyword_match": round(keyword_score, 2),
        "skills_match": round(skills_score, 2),
        "semantic_similarity": round(semantic_score, 2),
        "experience_match": round(experience_score, 2),
        "education_match": round(education_score, 2),
        "formatting_score": round(formatting_score, 2),
        "completeness_score": round(completeness_score, 2),
    }

    recommendations = generate_recommendations(breakdown, missing_keywords, mode)

    recommendations_by_section = None
    if mode == "with_jd":
        recommendations_by_section = {
            "Keywords": [
                "Add missing JD keywords naturally in Summary/Skills/Experience.",
                f"Prioritize: {', '.join(missing_keywords[:8])}" if missing_keywords else "No major keyword gaps detected.",
            ],
            "Skills": [
                "Use the JD tools/skills as Skills headings; reflect them in bullets.",
                "Add 6–10 JD skills across Skills + Experience + Projects.",
            ],
            "Experience": [
                "Use action + tool/process + impact bullets; mirror JD responsibilities.",
                "Make date ranges explicit (Month YYYY – Month YYYY).",
            ],
            "Formatting": [
                "Single-column ATS-friendly layout; no tables/text boxes.",
                "Consistent headers and bullet formatting.",
            ],
        }

    return {
        "overall_score": round(float(overall_score), 2),
        "breakdown": breakdown,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "detected_skills": detected_skills,
        "matched_skills": matched_skills,
        "recommendations": recommendations,
        "mode": mode,
        "warnings": warnings,
        "score_version": score_version,
        "embedding_model": embedding_model,
        "similarity_score": similarity_score,
        "recommendations_by_section": recommendations_by_section,
    }