
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768

DUPLICATE_THRESHOLD = 0.92
POSSIBLE_DUPLICATE_THRESHOLD = 0.82
MIN_KEYWORD_SCORE = 0.15
MIN_COMBINED_SCORE = 0.72

IDEAS_DIR = Path("ideas")
INDEX_FILE = IDEAS_DIR / "index.json"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
ISSUE_NUMBER = os.environ.get("ISSUE_NUMBER", "")

GITHUB_API_URL = "https://api.github.com"

STOPWORDS = {
    "about", "after", "again", "also", "among", "another",
    "because", "been", "before", "being", "between", "both",
    "could", "does", "doing", "each", "from", "have", "having",
    "into", "itself", "just", "more", "most", "other", "over",
    "same", "should", "some", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "very", "want", "well", "were", "what",
    "when", "where", "which", "while", "will", "with", "would",
    "your", "you", "idea", "ideas", "problem", "solution",
    "benefit", "benefits"
}


# --------------------------------------------------
# GitHub API helpers
# --------------------------------------------------

def github_request(method, endpoint, payload=None):
    """Make an authenticated GitHub API request."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN environment variable is missing.")

    url = (
        endpoint
        if endpoint.startswith("https://")
        else f"{GITHUB_API_URL}{endpoint}"
    )

    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "idea-submission-processor",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API request failed ({exc.code}): {error_body}"
        ) from exc


def add_issue_comment(message):
    """Post a comment on the current issue."""
    github_request(
        "POST",
        f"/repos/{GITHUB_REPOSITORY}/issues/{ISSUE_NUMBER}/comments",
        {"body": message},
    )


def add_issue_labels(*labels):
    """Add labels to the current issue."""
    github_request(
        "POST",
        f"/repos/{GITHUB_REPOSITORY}/issues/{ISSUE_NUMBER}/labels",
        {"labels": list(labels)},
    )


# --------------------------------------------------
# Issue parsing
# --------------------------------------------------

def get_field(body, field_name):
    """Extract a field from a GitHub issue-form body."""
    pattern = (
        r"^###\s+"
        + re.escape(field_name)
        + r"\s*\n(.*?)(?=^###\s+|\Z)"
    )

    match = re.search(
        pattern,
        body or "",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    if not match:
        return ""

    value = match.group(1).strip()

    if value.lower() in {
        "_no response_",
        "no response",
        "_none_",
    }:
        return ""

    return value


def normalize_issue_title(title):
    """Remove common issue-form title prefixes."""
    return re.sub(
        r"^\s*\[IDEA\]\s*:?\s*",
        "",
        title or "",
        flags=re.IGNORECASE,
    ).strip()


# --------------------------------------------------
# Text and keyword processing
# --------------------------------------------------

def extract_keywords(text):
    """Extract normalized keywords for lexical comparison."""
    words = re.findall(r"[a-zA-Z0-9+#.-]+", (text or "").lower())

    return sorted({
        word
        for word in words
        if len(word) >= 4 and word not in STOPWORDS
    })


def keyword_similarity(keywords_a, keywords_b):
    """Calculate Jaccard similarity between two keyword collections."""
    set_a = set(keywords_a or [])
    set_b = set(keywords_b or [])

    if not set_a or not set_b:
        return 0.0

    union = set_a | set_b

    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


def build_full_semantic_text(
    title,
    problem,
    solution,
    benefit,
    audience,
    category,
    additional,
):
    """Build the complete text used for keyword extraction."""
    return "\n".join(
        part.strip()
        for part in [
            f"Title: {title}",
            f"Problem: {problem}",
            f"Proposed solution: {solution}",
            f"Expected benefit: {benefit}",
            f"Target audience: {audience}",
            f"Category: {category}",
            f"Additional details: {additional}",
        ]
        if part and part.strip()
    )


def build_semantic_text(
    title,
    problem,
    solution,
    benefit,
    audience,
    category,
):
    """Build a consistent text representation for embeddings."""
    return "\n".join([
        f"Title: {title}",
        f"Problem: {problem}",
        f"Proposed solution: {solution}",
        f"Expected benefit: {benefit}",
        f"Target audience: {audience}",
        f"Category: {category}",
    ])


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

def generate_embedding(text):
    """Generate an embedding using the Gemini API."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

    model = urllib.parse.quote(EMBEDDING_MODEL, safe="-")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:embedContent?key="
        f"{urllib.parse.quote(GEMINI_API_KEY, safe='')}"
    )

    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {
            "parts": [{"text": text}],
        },
        "outputDimensionality": EMBEDDING_DIMENSIONS,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "idea-submission-processor",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini embedding request failed ({exc.code}): {error_body}"
        ) from exc

    embedding = result.get("embedding", {}).get("values")

    if not isinstance(embedding, list):
        raise RuntimeError(
            "Gemini API response did not contain embedding values."
        )

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"expected {EMBEDDING_DIMENSIONS}, got {len(embedding)}."
        )

    return embedding


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity for two embedding vectors."""
    if not isinstance(vector_a, list) or not isinstance(vector_b, list):
        return 0.0

    if not vector_a or len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# --------------------------------------------------
# Index and file helpers
# --------------------------------------------------

def load_index():
    """Load the existing idea index."""
    if not INDEX_FILE.exists():
        return []

    with INDEX_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise RuntimeError(
            f"{INDEX_FILE} must contain a JSON array of ideas."
        )

    return data


def save_index(ideas):
    """Write the updated idea index."""
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = INDEX_FILE.with_suffix(".json.tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(ideas, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary_file.replace(INDEX_FILE)


def next_idea_id(ideas):
    """Choose the next numeric idea ID."""
    numeric_ids = []

    for idea in ideas:
        try:
            numeric_ids.append(int(idea.get("id", 0)))
        except (TypeError, ValueError):
            continue

    return max(numeric_ids, default=0) + 1


def create_idea_file(
    idea_id,
    issue_number,
    username,
    title,
    category,
    submitted_at,
    keywords,
    problem,
    solution,
    benefit,
    audience,
    additional,
):
    """Create a Markdown file for an accepted idea."""
    IDEAS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON-quoted strings are safe YAML quoted scalars.
    # This avoids backslashes inside f-string expressions.
    escaped_username = json.dumps(username, ensure_ascii=False)
    escaped_title = json.dumps(title, ensure_ascii=False)
    escaped_category = json.dumps(category, ensure_ascii=False)

    keyword_text = json.dumps(keywords, ensure_ascii=False)

    content = f"""---
id: {idea_id}
issue_number: {issue_number}
username: {escaped_username}
title: {escaped_title}
category: {escaped_category}
submitted_at: {json.dumps(submitted_at)}
keywords: {keyword_text}
---

# {title}

## Problem / Pain Point

{problem or "Not provided."}

## Proposed Solution

{solution or "Not provided."}

## Expected Benefit

{benefit or "Not provided."}

## Who Would Benefit?

{audience or "Not provided."}

## Additional Details

{additional or "Not provided."}

## Submission

- GitHub issue: #{issue_number}
- Submitted by: {username}
- Submitted at: {submitted_at}
"""

    idea_path = IDEAS_DIR / f"idea-{idea_id}.md"
    idea_path.write_text(content, encoding="utf-8")

    return idea_path


# --------------------------------------------------
# Duplicate detection
# --------------------------------------------------

def find_best_match(new_embedding, new_keywords, ideas):
    """Find the existing idea with the strongest combined match."""
    best_match = None
    best_combined_score = -1.0

    for idea in ideas:
        existing_embedding = idea.get("embedding")

        # Skip entries that cannot be compared semantically.
        if not isinstance(existing_embedding, list):
            continue

        semantic_score = cosine_similarity(
            new_embedding,
            existing_embedding,
        )

        keyword_score = keyword_similarity(
            new_keywords,
            idea.get("keywords", []),
        )

        combined_score = (
            semantic_score * 0.7
            + keyword_score * 0.3
        )

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_match = {
                "idea": idea,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score,
            }

    return best_match


def classify_match(match):
    """Return duplicate, possible_duplicate, or None."""
    if not match:
        return None

    semantic = match["semantic_score"]
    keyword = match["keyword_score"]
    combined = match["combined_score"]

    if (
        semantic >= DUPLICATE_THRESHOLD
        and keyword >= MIN_KEYWORD_SCORE
        and combined >= MIN_COMBINED_SCORE
    ):
        return "duplicate"

    if (
        semantic >= POSSIBLE_DUPLICATE_THRESHOLD
        and keyword >= MIN_KEYWORD_SCORE
        and combined >= MIN_COMBINED_SCORE
    ):
        return "possible_duplicate"

    return None


def format_match_details(match):
    """Format the matching idea and scores for an issue comment."""
    idea = match["idea"]

    existing_title = idea.get("title", "Untitled idea")
    existing_id = idea.get("id", "unknown")

    return (
        f"**Closest existing idea:** {existing_title} "
        f"(ID: {existing_id})\n\n"
        f"- Semantic similarity: "
        f"{match['semantic_score']:.3f}\n"
        f"- Keyword similarity: "
        f"{match['keyword_score']:.3f}\n"
        f"- Combined score: "
        f"{match['combined_score']:.3f}"
    )


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main():
    if not GITHUB_REPOSITORY:
        raise RuntimeError(
            "GITHUB_REPOSITORY environment variable is missing."
        )

    if not ISSUE_NUMBER:
        raise RuntimeError(
            "ISSUE_NUMBER environment variable is missing."
        )

    issue_number = int(ISSUE_NUMBER)

    issue = github_request(
        "GET",
        f"/repos/{GITHUB_REPOSITORY}/issues/{issue_number}",
    )

    issue_body = issue.get("body") or ""
    username = (
        issue.get("user", {}).get("login")
        or "unknown"
    )
    title = normalize_issue_title(issue.get("title", ""))

    problem = get_field(issue_body, "Problem / Pain Point")
    solution = get_field(issue_body, "Proposed Solution")
    benefit = get_field(issue_body, "Expected Benefit")
    audience = get_field(issue_body, "Who Would Benefit?")
    category = get_field(issue_body, "Category")
    additional = get_field(issue_body, "Additional Details")

    required_fields = {
        "Idea Title": title,
        "Problem / Pain Point": problem,
        "Proposed Solution": solution,
        "Expected Benefit": benefit,
        "Who Would Benefit?": audience,
        "Category": category,
    }

    missing_fields = [
        name for name, value in required_fields.items()
        if not value.strip()
    ]

    if missing_fields:
        add_issue_comment(
            "I couldn't process this idea because these required "
            "fields are missing or empty:\n\n"
            + "\n".join(f"- {field}" for field in missing_fields)
        )
        return

    full_text = build_full_semantic_text(
        title,
        problem,
        solution,
        benefit,
        audience,
        category,
        additional,
    )

    keywords = extract_keywords(full_text)

    semantic_text = build_semantic_text(
        title,
        problem,
        solution,
        benefit,
        audience,
        category,
    )

    print(f"Generating embedding for issue #{issue_number}...")
    embedding = generate_embedding(semantic_text)

    ideas = load_index()

    match = find_best_match(embedding, keywords, ideas)
    match_type = classify_match(match)

    if match_type == "duplicate":
        add_issue_comment(
            "## Duplicate idea detected\n\n"
            "This submission appears to be very similar to an "
            "existing idea. It has **not** been added to the idea "
            "index.\n\n"
            f"{format_match_details(match)}\n\n"
            "If this idea is meaningfully different, please explain "
            "the distinction in the issue."
        )
        add_issue_labels("duplicate")
        print("Strong duplicate detected; not storing this idea.")
        return

    if match_type == "possible_duplicate":
        add_issue_comment(
            "## Possible duplicate idea\n\n"
            "This submission resembles an existing idea, but it "
            "will still be recorded for review.\n\n"
            f"{format_match_details(match)}\n\n"
            "Please review the existing idea and clarify what is "
            "different about this submission if needed."
        )
        add_issue_labels("possible-duplicate")

    idea_id = next_idea_id(ideas)
    submitted_at = datetime.now(timezone.utc).isoformat()

    idea_path = create_idea_file(
        idea_id=idea_id,
        issue_number=issue_number,
        username=username,
        title=title,
        category=category,
        submitted_at=submitted_at,
        keywords=keywords,
        problem=problem,
        solution=solution,
        benefit=benefit,
        audience=audience,
        additional=additional,
    )

    new_idea = {
        "id": idea_id,
        "issue_number": issue_number,
        "username": username,
        "title": title,
        "category": category,
        "submitted_at": submitted_at,
        "keywords": keywords,
        "embedding": embedding,
    }

    ideas.append(new_idea)
    save_index(ideas)

    add_issue_comment(
        "## Idea recorded successfully\n\n"
        f"Your idea has been saved as **Idea #{idea_id}: {title}**.\n\n"
        f"- Category: {category}\n"
        f"- File: `{idea_path.as_posix()}`\n"
        "- Status: recorded for review"
    )

    add_issue_labels("accepted")

    print(
        f"Recorded idea #{idea_id} from issue #{issue_number} "
        f"at {idea_path}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
