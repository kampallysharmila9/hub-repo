#!/usr/bin/env python3

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-2-preview"

EMBEDDING_DIMENSIONS = 768

# Strong enough similarity to automatically treat the
# submission as a duplicate.
DUPLICATE_THRESHOLD = 0.90

# Similar enough to alert the user, but not reject.
POSSIBLE_DUPLICATE_THRESHOLD = 0.75


# ============================================================
# ENVIRONMENT
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
ISSUE_NUMBER = os.environ.get("ISSUE_NUMBER", "")


# ============================================================
# PATHS
# ============================================================

INDEX_PATH = Path("ideas/index.json")
IDEAS_DIRECTORY = Path("ideas")


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "will",
    "would",
    "should",
    "about",
    "which",
    "there",
    "their",
    "into",
    "also",
    "when",
    "where",
    "what",
    "while",
    "your",
    "than",
    "them",
    "they",
    "been",
    "were",
    "being",
    "could",
    "very",
    "more",
    "some",
    "want",
    "like",
    "idea",
    "user",
    "users",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_json(path):
    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        fail(f"Could not parse {path}: {error}")


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )


# ============================================================
# GITHUB API
# ============================================================

def github_request(method, endpoint, payload=None):

    if not GITHUB_TOKEN:
        fail("GITHUB_TOKEN is missing.")

    if not GITHUB_REPOSITORY:
        fail("GITHUB_REPOSITORY is missing.")

    url = (
        "https://api.github.com"
        f"/repos/{GITHUB_REPOSITORY}/{endpoint}"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "idea-hub",
    }

    body = None

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

        headers["Content-Type"] = (
            "application/json"
        )

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method
    )

    try:

        with urllib.request.urlopen(request) as response:

            content = (
                response
                .read()
                .decode("utf-8")
            )

            if not content:
                return {}

            return json.loads(content)

    except urllib.error.HTTPError as error:

        response_body = (
            error
            .read()
            .decode("utf-8")
        )

        fail(
            f"GitHub API error "
            f"{error.code}: "
            f"{response_body}"
        )

    except urllib.error.URLError as error:

        fail(
            f"GitHub connection error: {error}"
        )


def get_issue():

    return github_request(
        "GET",
        f"issues/{ISSUE_NUMBER}"
    )


def create_issue_comment(message):

    return github_request(
        "POST",
        f"issues/{ISSUE_NUMBER}/comments",
        {
            "body": message
        }
    )


def add_issue_label(label):

    return github_request(
        "POST",
        f"issues/{ISSUE_NUMBER}/labels",
        {
            "labels": [label]
        }
    )


# ============================================================
# ISSUE FORM PARSING
# ============================================================

def get_field(body, label):

    escaped_label = re.escape(label)

    pattern = (
        rf"###\s*{escaped_label}"
        rf"\s*\n+"
        rf"([\s\S]*?)"
        rf"(?=\n###|\Z)"
    )

    match = re.search(
        pattern,
        body,
        re.IGNORECASE
    )

    if not match:
        return ""

    value = match.group(1).strip()

    if value.startswith("_No response_"):
        return ""

    return value


# ============================================================
# KEYWORDS
# ============================================================

def extract_keywords(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    result = []

    for word in words:

        if len(word) < 4:
            continue

        if word in STOPWORDS:
            continue

        if word not in result:
            result.append(word)

    return result


# ============================================================
# JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(a, b):

    set_a = set(a or [])
    set_b = set(b or [])

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a & set_b
    )

    union = len(
        set_a | set_b
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    if not isinstance(a, list):
        return 0.0

    if not isinstance(b, list):
        return 0.0

    if len(a) != len(b):
        return 0.0

    dot = 0.0
    magnitude_a = 0.0
    magnitude_b = 0.0

    for x, y in zip(a, b):

        dot += x * y
        magnitude_a += x * x
        magnitude_b += y * y

    if magnitude_a == 0:
        return 0.0

    if magnitude_b == 0:
        return 0.0

    return dot / (
        math.sqrt(magnitude_a)
        *
        math.sqrt(magnitude_b)
    )


# ============================================================
# GEMINI EMBEDDING
# ============================================================

def generate_embedding(text):

    if not GEMINI_API_KEY:
        fail(
            "GEMINI_API_KEY is missing."
        )

    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{EMBEDDING_MODEL}:embedContent"
    )

    payload = {
        "model":
            f"models/{EMBEDDING_MODEL}",

        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        },

        "output_dimensionality":
            EMBEDDING_DIMENSIONS
    }

    request = urllib.request.Request(
        url,

        data=json.dumps(
            payload
        ).encode("utf-8"),

        headers={
            "Content-Type":
                "application/json",

            "x-goog-api-key":
                GEMINI_API_KEY
        },

        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request
        ) as response:

            data = json.loads(
                response
                .read()
                .decode("utf-8")
            )

    except urllib.error.HTTPError as error:

        response_body = (
            error
            .read()
            .decode("utf-8")
        )

        fail(
            f"Gemini API error "
            f"{error.code}: "
            f"{response_body}"
        )

    except urllib.error.URLError as error:

        fail(
            f"Gemini connection error: "
            f"{error}"
        )

    embedding = (
        data
        .get("embedding", {})
        .get("values")
    )

    if not embedding:
        fail(
            "Gemini returned no embedding."
        )

    return embedding


# ============================================================
# SEMANTIC TEXT
# ============================================================

def build_semantic_text(
    title,
    problem,
    solution,
    benefit,
    users,
    category,
    additional
):

    return f"""
Idea Title:
{title}

Problem:
{problem}

Proposed Solution:
{solution}

Expected Benefit:
{benefit}

Who Would Benefit:
{users}

Category:
{category}

Additional Details:
{additional}
""".strip()


# ============================================================
# SLUGIFY
# ============================================================

def slugify(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s-]",
        "",
        text
    )

    words = text.strip().split()

    return "-".join(words[:8]) or "idea"


# ============================================================
# YAML ESCAPING
# ============================================================

def yaml_string(value):

    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )


# ============================================================
# CREATE IDEA FILE
# ============================================================

def create_idea_file(
    idea_id,
    name,
    title,
    category,
    problem,
    solution,
    benefit,
    users,
    additional,
    keywords,
    issue_number,
    submitted_at
):

    filename = (
        f"{idea_id:04d}-"
        f"{slugify(title)}.md"
    )

    path = (
        IDEAS_DIRECTORY / filename
    )

    keyword_text = ", ".join(
        f'"{yaml_string(k)}"'
        for k in keywords
    )

    content = f"""---
id: {idea_id}
issue_number: {issue_number}
name: "{yaml_string(name)}"
title: "{yaml_string(title)}"
category: "{yaml_string(category)}"
submitted_at: "{submitted_at}"
keywords: [{keyword_text}]
---

# {title}

## Problem / Pain Point

{problem}

## Proposed Solution

{solution}

## Expected Benefit

{benefit}

## Who Would Benefit?

{users}

## Category

{category}

## Additional Details

{additional or "None provided."}

## Original Issue

#{issue_number}
"""

    IDEAS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content,
        encoding="utf-8"
    )

    return str(path)


# ============================================================
# DUPLICATE REPORT
# ============================================================

def report_duplicate(
    match,
    semantic_score,
    keyword_score
):

    existing_title = (
        match.get(
            "title",
            match.get(
                "file",
                "Existing idea"
            )
        )
    )

    message = f"""## 🔴 Possible Duplicate

Your idea appears very similar to an existing submission.

### Existing idea

**{existing_title}**

Issue: #{match.get("issue_number", "unknown")}

File: `{match.get("file", "unknown")}`

### Similarity

- Semantic similarity: **{semantic_score * 100:.1f}%**
- Keyword similarity: **{keyword_score * 100:.1f}%**

Please review the existing idea before continuing.

If you believe your idea is genuinely different, explain the difference in a comment so a maintainer can review it.
"""

    create_issue_comment(message)

    add_issue_label(
        "possible-duplicate"
    )


# ============================================================
# POSSIBLE DUPLICATE
# ============================================================

def report_possible_duplicate(
    match,
    semantic_score,
    keyword_score
):

    existing_title = (
        match.get(
            "title",
            match.get(
                "file",
                "Existing idea"
            )
        )
    )

    message = f"""## 🟡 Similar Idea Found

Your idea appears related to an existing submission.

### Existing idea

**{existing_title}**

Issue: #{match.get("issue_number", "unknown")}

File: `{match.get("file", "unknown")}`

### Similarity

- Semantic similarity: **{semantic_score * 100:.1f}%**
- Keyword similarity: **{keyword_score * 100:.1f}%**

This is not being automatically rejected. Please explain why your idea is different if appropriate.
"""

    create_issue_comment(message)

    add_issue_label(
        "possible-duplicate"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IDEA HUB")
    print("Processing idea")
    print("=" * 60)

    if not ISSUE_NUMBER:
        fail("ISSUE_NUMBER is missing.")

    print(
        f"Processing issue #{ISSUE_NUMBER}"
    )

    # --------------------------------------------------------
    # Get issue
    # --------------------------------------------------------

    issue = get_issue()

    body = (
        issue.get("body")
        or ""
    )

    if not body:
        fail("Issue body is empty.")

    # --------------------------------------------------------
    # Extract form fields
    # --------------------------------------------------------

    name = get_field(
        body,
        "Your Name"
    )

    title = get_field(
        body,
        "Idea Title"
    )

    problem = get_field(
        body,
        "Problem / Pain Point"
    )

    solution = get_field(
        body,
        "Proposed Solution"
    )

    benefit = get_field(
        body,
        "Expected Benefit"
    )

    users = get_field(
        body,
        "Who Would Benefit?"
    )

    category = get_field(
        body,
        "Category"
    )

    additional = get_field(
        body,
        "Additional Details"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required = {
        "Your Name": name,
        "Idea Title": title,
        "Problem / Pain Point": problem,
        "Proposed Solution": solution,
        "Expected Benefit": benefit,
        "Who Would Benefit?": users,
        "Category": category,
    }

    missing = [
        field
        for field, value in required.items()
        if not value
    ]

    if missing:

        fail(
            "Missing required fields: "
            + ", ".join(missing)
        )

    print("All required fields found.")

    # --------------------------------------------------------
    # Build semantic text
    # --------------------------------------------------------

    semantic_text = build_semantic_text(
        title,
        problem,
        solution,
        benefit,
        users,
        category,
        additional
    )

    # --------------------------------------------------------
    # Keywords
    # --------------------------------------------------------

    keywords = extract_keywords(
        semantic_text
    )

    print(
        f"Extracted {len(keywords)} keywords."
    )

    # --------------------------------------------------------
    # Generate embedding
    # --------------------------------------------------------

    print(
        "Generating Gemini embedding..."
    )

    embedding = generate_embedding(
        semantic_text
    )

    print(
        f"Generated {len(embedding)} dimensions."
    )

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    index = load_json(
        INDEX_PATH
    )

    if not index:
        index = {
            "ideas": []
        }

    if "ideas" not in index:
        index["ideas"] = []

    # --------------------------------------------------------
    # Find closest existing idea
    # --------------------------------------------------------

    best_match = None
    best_semantic_score = 0.0
    best_keyword_score = 0.0

    for entry in index["ideas"]:

        existing_embedding = (
            entry.get("embedding")
        )

        if not isinstance(
            existing_embedding,
            list
        ):

            print(
                f"Skipping idea "
                f"#{entry.get('id')} "
                f"- no embedding."
            )

            continue

        semantic_score = (
            cosine_similarity(
                embedding,
                existing_embedding
            )
        )

        keyword_score = (
            jaccard_similarity(
                keywords,
                entry.get(
                    "keywords",
                    []
                )
            )
        )

        print(
            f"Idea #{entry.get('id')}: "
            f"semantic={semantic_score:.4f}, "
            f"keywords={keyword_score:.4f}"
        )

        if (
            semantic_score
            > best_semantic_score
        ):

            best_semantic_score = (
                semantic_score
            )

            best_keyword_score = (
                keyword_score
            )

            best_match = entry

    # --------------------------------------------------------
    # Decide duplicate status
    # --------------------------------------------------------

    if best_match:

        print("")
        print(
            f"Best match: "
            f"#{best_match.get('id')}"
        )

        print(
            f"Semantic similarity: "
            f"{best_semantic_score:.4f}"
        )

        print(
            f"Keyword similarity: "
            f"{best_keyword_score:.4f}"
        )

        if (
            best_semantic_score
            >= DUPLICATE_THRESHOLD
        ):

            print(
                "Strong duplicate detected."
            )

            report_duplicate(
                best_match,
                best_semantic_score,
                best_keyword_score
            )

            print(
                "Idea was not stored."
            )

            return

        elif (
            best_semantic_score
            >= POSSIBLE_DUPLICATE_THRESHOLD
        ):

            print(
                "Possible duplicate detected."
            )

            report_possible_duplicate(
                best_match,
                best_semantic_score,
                best_keyword_score
            )

    else:

        print(
            "No existing embeddings found."
        )

    # --------------------------------------------------------
    # Generate next ID
    # --------------------------------------------------------

    ids = []

    for entry in index["ideas"]:

        try:
            ids.append(
                int(entry.get("id", 0))
            )
        except (
            ValueError,
            TypeError
        ):
            pass

    next_id = (
        max(ids)
        if ids
        else 0
    ) + 1

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    submitted_at = (
        datetime
        .now(timezone.utc)
        .isoformat()
    )

    # --------------------------------------------------------
    # Write Markdown file
    # --------------------------------------------------------

    file_path = create_idea_file(
        next_id,
        name,
        title,
        category,
        problem,
        solution,
        benefit,
        users,
        additional,
        keywords,
        ISSUE_NUMBER,
        submitted_at
    )

    print(
        f"Created {file_path}"
    )

    # --------------------------------------------------------
    # Update index
    # --------------------------------------------------------

    index["ideas"].append({

        "id":
            next_id,

        "issue_number":
            int(ISSUE_NUMBER),

        "file":
            file_path,

        "name":
            name,

        "title":
            title,

        "category":
            category,

        "problem":
            problem,

        "solution":
            solution,

        "benefit":
            benefit,

        "users":
            users,

        "keywords":
            keywords,

        "embedding":
            embedding,

        "submitted_at":
            submitted_at
    })

    save_json(
        INDEX_PATH,
        index
    )

    # --------------------------------------------------------
    # GitHub comment
    # --------------------------------------------------------

    create_issue_comment(
        f"""## ✅ Idea Recorded

Thank you{f", {name}" if name else ""}!

Your idea has been recorded as:

**{title}**

**Category:** {category}

Stored at:

`{file_path}`

The submission was also checked against existing ideas using semantic similarity.
"""
    )

    add_issue_label(
        "accepted"
    )

    print("")
    print("=" * 60)
    print(
        f"Successfully stored idea #{next_id}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
