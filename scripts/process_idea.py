
#!/usr/bin/env python3
"""Process a submitted GitHub idea using Gemini embeddings and cosine similarity only."""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from embedding_utils import (
    cosine_similarity,
    generate_embedding,
    idea_text,
    load_index,
    save_index,
)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
ISSUE_NUMBER = os.environ.get("ISSUE_NUMBER", "")

DUPLICATE_THRESHOLD = float(
    os.environ.get("DUPLICATE_THRESHOLD", "0.82")
)
POSSIBLE_DUPLICATE_THRESHOLD = float(
    os.environ.get("POSSIBLE_DUPLICATE_THRESHOLD", "0.70")
)

IDEAS_DIR = Path("ideas")


def github_request(method, endpoint, payload=None):
    """Send a request to the GitHub REST API."""
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        raise RuntimeError(
            "GITHUB_TOKEN and GITHUB_REPOSITORY are required."
        )

    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_REPOSITORY}/{endpoint.lstrip('/')}"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gemini-embedding-idea-processor",
    }

    data = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )

    if data is not None:
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
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API error {exc.code}: {body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"GitHub connection error: {exc}"
        ) from exc


def comment(message):
    """Add a comment to the submitted issue."""
    github_request(
        "POST",
        f"issues/{ISSUE_NUMBER}/comments",
        {"body": message},
    )


def labels(*values):
    """Add one or more labels to the submitted issue."""
    github_request(
        "POST",
        f"issues/{ISSUE_NUMBER}/labels",
        {"labels": list(values)},
    )


def get_field(body, field_name):
    """Extract a field from a GitHub issue form."""
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


def normalize_title(title):
    """Remove an optional [IDEA] prefix."""
    return re.sub(
        r"^\s*\[IDEA\]\s*:?[ \t]*",
        "",
        title or "",
        flags=re.IGNORECASE,
    ).strip()


def next_id(ideas):
    """Return the next numeric idea ID."""
    ids = []

    for idea in ideas:
        try:
            ids.append(int(idea.get("id", 0)))
        except (TypeError, ValueError):
            continue

    return max(ids, default=0) + 1


def create_idea_file(idea):
    """Write an idea to Markdown and return its path."""
    IDEAS_DIR.mkdir(parents=True, exist_ok=True)

    path = IDEAS_DIR / f"idea-{idea['id']}.md"

    content = f"""---
id: {idea['id']}
issue_number: {idea['issue_number']}
username: {json.dumps(idea['username'], ensure_ascii=False)}
title: {json.dumps(idea['title'], ensure_ascii=False)}
category: {json.dumps(idea['category'], ensure_ascii=False)}
submitted_at: {json.dumps(idea['submitted_at'])}
---

# {idea['title']}

## Problem / Pain Point

{idea['problem'] or 'Not provided.'}

## Proposed Solution

{idea['solution'] or 'Not provided.'}

## Expected Benefit

{idea['benefit'] or 'Not provided.'}

## Who Would Benefit?

{idea['audience'] or 'Not provided.'}

## Additional Details

{idea['additional'] or 'Not provided.'}
"""

    path.write_text(content, encoding="utf-8")
    return path


def main():
    if not ISSUE_NUMBER:
        raise RuntimeError("ISSUE_NUMBER is missing.")

    issue = github_request("GET", f"issues/{ISSUE_NUMBER}")

    title = normalize_title(issue.get("title", ""))
    body = issue.get("body") or ""
    username = (issue.get("user") or {}).get("login") or "unknown"

    new_idea = {
        "title": title,
        "problem": get_field(body, "Problem / Pain Point"),
        "solution": get_field(body, "Proposed Solution"),
        "benefit": get_field(body, "Expected Benefit"),
        "audience": get_field(body, "Who Would Benefit?"),
        "category": get_field(body, "Category"),
        "additional": get_field(body, "Additional Details"),
    }

    required = [
        "title",
        "problem",
        "solution",
        "benefit",
        "audience",
        "category",
    ]

    missing = [
        name
        for name in required
        if not str(new_idea.get(name, "")).strip()
    ]

    if missing:
        comment(
            "I couldn't process this idea because these required "
            "fields are missing or empty:\n\n"
            + "\n".join(f"- {name}" for name in missing)
        )
        return

    ideas, index_shape = load_index()

    new_text = idea_text(new_idea)

    print(
        f"Loaded {len(ideas)} existing ideas. "
        "Generating new idea embedding..."
    )

    # Generate the new idea's Gemini embedding.
    new_embedding = generate_embedding(new_text)

    matches = []
    generated_old = 0

    # Compare embeddings only. There is no text-generation classifier,
    # cross-encoder, or keyword-based duplicate decision.
    for existing in ideas:
        embedding = existing.get("embedding")

        # Generate and save a missing embedding for an older idea.
        if not isinstance(embedding, list) or not embedding:
            existing_text = idea_text(existing)

            if not existing_text:
                print(
                    f"Warning: existing idea "
                    f"#{existing.get('id', '?')} has no text; skipping it.",
                    file=sys.stderr,
                )
                continue

            print(
                "Generating missing embedding for existing idea "
                f"#{existing.get('id', '?')}..."
            )

            embedding = generate_embedding(existing_text)
            existing["embedding"] = embedding
            generated_old += 1

        if len(embedding) != len(new_embedding):
            raise RuntimeError(
                f"Embedding dimensions differ for idea "
                f"#{existing.get('id', '?')}. "
                "Regenerate all embeddings using the same Gemini "
                "embedding model and dimensions."
            )

        # The duplicate score is cosine similarity between vectors.
        score = cosine_similarity(new_embedding, embedding)

        if score >= POSSIBLE_DUPLICATE_THRESHOLD:
            matches.append({
                "id": existing.get("id"),
                "title": existing.get("title", "Untitled idea"),
                "issue_number": existing.get("issue_number"),
                "similarity": score,
                "classification": (
                    "duplicate"
                    if score >= DUPLICATE_THRESHOLD
                    else "possible_duplicate"
                ),
            })

    # Persist embeddings generated for older ideas.
    if generated_old:
        save_index(ideas, index_shape)

    matches.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    definite = [
        match
        for match in matches
        if match["classification"] == "duplicate"
    ]

    # A definite duplicate is not added to the ideas index.
    if definite:
        lines = [
            "## Duplicate idea detected",
            "",
            "This submission has not been added to the ideas index.",
            "",
        ]

        for match in definite[:10]:
            lines.append(
                f"- **{match['title']}** "
                f"(Idea ID: {match['id']}, "
                f"Issue: #{match['issue_number']}; "
                f"cosine similarity: {match['similarity']:.3f})"
            )

        lines.extend([
            "",
            "Similarity is an automated screening signal. "
            "If this idea is meaningfully different, please explain "
            "the distinction in the issue.",
        ])

        comment("\n".join(lines))
        labels("duplicate")

        print(
            f"Duplicate detected: {len(definite)} match(es). "
            "Submission not recorded."
        )
        return

    # No definite duplicate: record the new idea and its embedding.
    idea_id = next_id(ideas)
    submitted_at = datetime.now(timezone.utc).isoformat()

    entry = {
        "id": idea_id,
        "issue_number": int(ISSUE_NUMBER),
        "username": username,
        "title": new_idea["title"],
        "category": new_idea["category"],
        "submitted_at": submitted_at,
        "problem": new_idea["problem"],
        "solution": new_idea["solution"],
        "benefit": new_idea["benefit"],
        "audience": new_idea["audience"],
        "additional": new_idea["additional"],
        "description": new_text,
        "embedding": new_embedding,
    }

    path = create_idea_file(entry)
    entry["file"] = path.as_posix()

    ideas.append(entry)
    save_index(ideas, index_shape)

    # A possible duplicate is still recorded for human review.
    if matches:
        lines = [
            "## Possible duplicate idea",
            "",
            "Your idea has been recorded for human review. "
            "Similar existing ideas:",
            "",
        ]

        for match in matches[:10]:
            lines.append(
                f"- **{match['title']}** "
                f"(Idea ID: {match['id']}, "
                f"Issue: #{match['issue_number']}; "
                f"cosine similarity: {match['similarity']:.3f}; "
                f"{match['classification']})"
            )

        comment("\n".join(lines))
        labels("possible-duplicate", "accepted")

    else:
        comment(
            "## Idea recorded successfully\n\n"
            f"Your idea has been saved as "
            f"**Idea #{idea_id}: {new_idea['title']}**.\n\n"
            f"- Category: {new_idea['category']}\n"
            f"- File: `{path.as_posix()}`\n"
            "- Duplicate check: Gemini embeddings + cosine similarity"
        )
        labels("accepted")

    print(
        f"Recorded idea #{idea_id}; "
        f"compared against {len(ideas) - 1} existing idea(s)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
