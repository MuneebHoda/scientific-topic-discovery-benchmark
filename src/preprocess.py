"""Text preprocessing helpers shared across the local-first pipeline."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


WHITESPACE_RE = re.compile(r"\s+")
INLINE_MATH_RE = re.compile(r"\$[^$]+\$")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?")
LATEX_ENV_RE = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PUNCT_RE = re.compile(r"[^\w\s]")
TITLE_HASH_PUNCT_RE = re.compile(r"[_\W]+")
TOKEN_SPLIT_RE = re.compile(r"\s+")

PLACEHOLDER_PHRASES = (
    "this paper has been withdrawn",
    "this article has been withdrawn",
    "this submission has been withdrawn",
    "this paper is withdrawn",
    "this paper was withdrawn",
    "the paper has been withdrawn",
    "withdrawn",
    "retracted",
)


def normalize_whitespace(text: object) -> str:
    """Collapse whitespace and coerce null-like values to empty strings."""

    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def parse_categories(raw_categories: object) -> List[str]:
    """Split the arXiv category string into ordered labels."""

    text = normalize_whitespace(raw_categories)
    return text.split() if text else []


def primary_category(categories: Sequence[str]) -> str:
    """Return the first category token, which acts as the primary label."""

    return categories[0] if categories else ""


def primary_domain(category: str) -> str:
    """Return the coarse domain derived from the primary category."""

    if not category:
        return ""
    return category.split(".", 1)[0]


def derive_text_input(title: str, abstract: str) -> str:
    """Combine title and abstract for downstream modeling."""

    return f"{normalize_whitespace(title)}\n{normalize_whitespace(abstract)}".strip()


def strip_latex_like_artifacts(text: str) -> str:
    """Remove the most disruptive LaTeX artifacts while preserving prose."""

    cleaned = normalize_whitespace(text)
    cleaned = LATEX_ENV_RE.sub(" ", cleaned)
    cleaned = INLINE_MATH_RE.sub(" ", cleaned)
    cleaned = LATEX_COMMAND_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    return normalize_whitespace(cleaned)


def normalized_hash_key(title: str, abstract: str) -> str:
    """Build the duplicate-content hash key aligned with the EDA spirit."""

    normalized_title = TITLE_HASH_PUNCT_RE.sub(" ", normalize_whitespace(title).lower())
    normalized_title = normalize_whitespace(normalized_title)
    normalized_abstract = normalize_whitespace(abstract).lower()
    return f"{normalized_title}\n{normalized_abstract}"


def abstract_word_len(abstract: str) -> int:
    """Count whitespace-delimited abstract tokens."""

    text = normalize_whitespace(abstract)
    if not text:
        return 0
    return len(text.split(" "))


def is_placeholder_abstract(abstract: str) -> bool:
    """Detect short withdrawn or retracted placeholder abstracts."""

    text = normalize_whitespace(abstract).lower()
    if not text:
        return True
    if len(text.split()) > 60:
        return False
    return any(phrase in text for phrase in PLACEHOLDER_PHRASES)


def submission_year_from_versions(versions: object) -> int | None:
    """Extract the submission year from the first version block."""

    from email.utils import parsedate_to_datetime

    if not isinstance(versions, list) or not versions:
        return None

    created = versions[0].get("created") if isinstance(versions[0], dict) else None
    if not created:
        return None

    try:
        return int(parsedate_to_datetime(created).year)
    except Exception:
        match = re.search(r"(19|20)\d{2}", str(created))
        return int(match.group(0)) if match else None


def update_year_from_value(value: object) -> int | None:
    """Extract the update year from the update_date field."""

    text = normalize_whitespace(value)
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def tfidf_clean(text: str) -> str:
    """Conservative sparse-text preprocessing used by TF-IDF."""

    cleaned = strip_latex_like_artifacts(text)
    cleaned = cleaned.lower()
    cleaned = PUNCT_RE.sub(" ", cleaned)
    tokens = [token for token in TOKEN_SPLIT_RE.split(cleaned) if token and token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def dense_clean(text: str) -> str:
    """Minimal text cleanup for dense encoders."""

    cleaned = normalize_whitespace(text)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = EMAIL_RE.sub(" ", cleaned)
    return normalize_whitespace(cleaned)


def tokenize_for_npmi(texts: Iterable[str]) -> List[List[str]]:
    """Tokenize a collection of texts using the TF-IDF cleaning rules."""

    tokenized = []
    for text in texts:
        cleaned = tfidf_clean(text)
        tokenized.append([token for token in cleaned.split() if token])
    return tokenized

