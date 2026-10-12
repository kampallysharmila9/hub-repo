"""Shared Gemini embedding and index helpers for idea duplicate detection."""
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
EMBEDDING_DIMENSIONS = int(os.environ.get("GEMINI_EMBEDDING_DIMENSIONS", "768"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
INDEX_PATH = Path("ideas/index.json")


def generate_embedding(text: str) -> list[float]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")
    if not text or not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent"
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text[:20000]}]},
        "output_dimensionality": EMBEDDING_DIMENSIONS,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini embedding API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc}") from exc
    values = data.get("embedding", {}).get("values")
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Gemini returned no embedding: {data}")
    if len(values) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(values)}. "
            "Use the same model and dimensions for all stored vectors."
        )
    return values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) ** 2 for x in a))
    norm_b = math.sqrt(sum(float(y) ** 2 for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def idea_text(idea: dict) -> str:
    fields = [
        ("Title", idea.get("title", "")),
        ("Problem", idea.get("problem", "")),
        ("Solution", idea.get("solution", "")),
        ("Benefit", idea.get("benefit", "")),
        ("Audience", idea.get("audience", "")),
        ("Category", idea.get("category", "")),
        ("Description", idea.get("description", "")),
        ("Additional details", idea.get("additional", idea.get("additional_details", ""))),
        ("Keywords", " ".join(idea.get("keywords", [])) if isinstance(idea.get("keywords"), list) else idea.get("keywords", "")),
    ]
    parts = [f"{label}: {value}" for label, value in fields if value and str(value).strip()]
    file_name = idea.get("file")
    idea_id = idea.get("id")
    candidates = []
    if file_name:
        candidates.append(Path(file_name))
    if idea_id is not None:
        candidates.append(Path("ideas") / f"idea-{idea_id}.md")
    for path in candidates:
        try:
            if path.is_file():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append("Idea document:\n" + content)
                    break
        except OSError:
            pass
    return "\n".join(parts).strip()


def load_index() -> tuple[list[dict], str]:
    """Return ideas and original shape ('wrapped' or 'list') to preserve schema."""
    if not INDEX_PATH.exists():
        raise RuntimeError(f"{INDEX_PATH} does not exist.")
    try:
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {INDEX_PATH}: {exc}") from exc
    if isinstance(raw, dict) and isinstance(raw.get("ideas"), list):
        return raw["ideas"], "wrapped"
    if isinstance(raw, list):
        return raw, "list"
    raise RuntimeError(f"{INDEX_PATH} must be a list of ideas or an object with an 'ideas' list.")


def save_index(ideas: list[dict], shape: str) -> None:
    if shape == "wrapped":
        raw = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {}
        raw["ideas"] = ideas
    else:
        raw = ideas
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(INDEX_PATH)
