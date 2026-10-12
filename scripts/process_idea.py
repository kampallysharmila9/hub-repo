
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash"

IDEAS_DIR = Path("ideas")
INDEX_FILE = IDEAS_DIR / "index.json"

# Gemini receives up to this many shortlisted ideas per request.
# Set to None to send every existing idea in one request.
# For a large repository, use a shortlist or process ideas in batches.
MAX_CANDIDATES = 40

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
    "your", "you", "idea", "ideas", "create", "build", "make",
    "using", "proposed", "expected", "benefit", "benefits",
}


# ============================================================
# GITHUB API
# ============================================================

def github_request(method, endpoint, payload=None):
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is missing.")

    url = (
        endpoint
        if endpoint.startswith("https://")
        else f"{GITHUB_API_URL}{endpoint}"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "idea-duplicate-checker",
    }

    data = None
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
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API error {exc.code}: {body}"
        ) from exc


def add_issue_comment(message):
    github_request(
        "POST",
        f"/repos/{GITHUB_REPOSITORY}/issues/{ISSUE_NUMBER}/comments",
        {"body": message},
    )


def add_issue_labels(*labels):
    github_request(
        "POST",
        f"/repos/{GITHUB_REPOSITORY}/issues/{ISSUE_NUMBER}/labels",
        {"labels": list(labels)},
    )


# ============================================================
# ISSUE FORM PARSING
# ============================================================

def get_field(body, field_name):
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
    return re.sub(
        r"^\s*\[IDEA\]\s*:?\s*",
        "",
        title or "",
        flags=re.IGNORECASE,
    ).strip()


# ============================================================
# OPTIONAL PYTHON KEYWORD SHORTLIST
# ============================================================

def normalize_text(text):
    return re.sub(
        r"[^a-z0-9+#]+",
        " ",
        str(text or "").lower(),
    ).strip()


def extract_keywords(text):
    words = re.findall(
        r"[a-zA-Z0-9+#]+",
        str(text or "").lower(),
    )

    return {
        word for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }


def lexical_similarity(text_a, text_b):
    """Return a lightweight score used only to rank candidates."""
    words_a = extract_keywords(text_a)
    words_b = extract_keywords(text_b)

    if words_a and words_b:
        jaccard = len(words_a & words_b) / len(words_a | words_b)
    else:
        jaccard = 0.0

    sequence = SequenceMatcher(
        None,
        normalize_text(text_a),
        normalize_text(text_b),
    ).ratio()

    return max(jaccard, sequence * 0.8)


def build_idea_text(idea):
    """Read all useful fields available in an index entry."""
    fields = [
        ("Title", idea.get("title", "")),
        ("Problem", idea.get("problem", "")),
        ("Solution", idea.get("solution", "")),
        ("Benefit", idea.get("benefit", "")),
        ("Audience", idea.get("audience", "")),
        ("Category", idea.get("category", "")),
        ("Description", idea.get("description", "")),
        ("Keywords", " ".join(idea.get("keywords", []))),
    ]

    parts = [
        f"{label}: {value}"
        for label, value in fields
        if value and str(value).strip()
    ]

    # Older index entries may not contain the full description.
    # Try loading the corresponding Markdown file if it exists.
    idea_id = idea.get("id")
    if idea_id is not None:
        markdown_path = IDEAS_DIR / f"idea-{idea_id}.md"

        if markdown_path.exists():
            try:
                markdown = markdown_path.read_text(encoding="utf-8")
                if markdown.strip():
                    parts.append("Existing idea document:\n" + markdown)
            except OSError as exc:
                print(
                    f"Warning: couldn't read {markdown_path}: {exc}",
                    file=sys.stderr,
                )

    return "\n".join(parts)


def shortlist_candidates(new_idea_text, ideas):
    """
    Rank ideas lexically, but do not make the lexical score
    the final duplicate decision.
    """
    ranked = []

    for idea in ideas:
        existing_text = build_idea_text(idea)
        score = lexical_similarity(new_idea_text, existing_text)

        ranked.append({
            "idea": idea,
            "text": existing_text,
            "shortlist_score": score,
        })

    ranked.sort(
        key=lambda item: item["shortlist_score"],
        reverse=True,
    )

    if MAX_CANDIDATES is None:
        return ranked

    return ranked[:MAX_CANDIDATES]


# ============================================================
# GEMINI TEXT-GENERATION API
# ============================================================

def call_gemini_for_duplicates(new_idea, candidates):
    """
    Ask Gemini to compare meaning, problem, and solution.

    No embeddings are generated or used.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    if not candidates:
        return {
            "decision": "unique",
            "matches": [],
            "summary": "No existing ideas were available to compare.",
        }

    model = urllib.parse.quote(GEMINI_MODEL, safe="-")
    api_key = urllib.parse.quote(GEMINI_API_KEY, safe="")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    candidate_payload = []

    for candidate in candidates:
        idea = candidate["idea"]

        candidate_payload.append({
            "id": str(idea.get("id", "unknown")),
            "title": str(idea.get("title", "Untitled idea")),
            "category": str(idea.get("category", "")),
            "content": candidate["text"][:6000],
        })

    prompt_data = {
        "new_idea": {
            "title": new_idea["title"],
            "problem": new_idea["problem"],
            "solution": new_idea["solution"],
            "benefit": new_idea["benefit"],
            "audience": new_idea["audience"],
            "category": new_idea["category"],
            "additional_details": new_idea["additional"],
        },
        "existing_ideas": candidate_payload,
    }

    prompt = """
You are a careful duplicate-idea reviewer for a GitHub idea repository.

Compare the new idea with every existing idea supplied in the JSON.

Judge conceptual similarity, not just matching words. Consider:
1. Whether they solve the same underlying problem.
2. Whether they propose essentially the same solution or capability.
3. Whether their intended users and expected outcomes overlap.
4. Whether differences are meaningful or merely changes in wording.

Important rules:
- Different wording can still describe a duplicate.
- Shared generic words alone do not prove a duplicate.
- Two ideas in the same category are not automatically duplicates.
- A dashboard for the same analytics purpose may be related even if
  one mentions user statistics and another system analytics.
- If evidence is insufficient, use possible_duplicate rather than
  asserting a definite duplicate.
- Compare against every supplied existing idea.
- Do not invent existing ideas or IDs.
- Return valid JSON only, matching this schema:

{
  "decision": "duplicate" | "possible_duplicate" | "unique",
  "matches": [
    {
      "id": "existing idea ID",
      "title": "existing idea title",
      "classification": "duplicate" | "possible_duplicate",
      "confidence": 0.0,
      "reason": "short explanation of the shared or different concepts"
    }
  ],
  "summary": "brief explanation"
}

Decision rules:
- "duplicate" if at least one match is clearly the same idea.
- "possible_duplicate" if no definite duplicate exists but at least
  one existing idea is substantially related.
- "unique" only if none of the supplied ideas is meaningfully similar.
- Include every relevant match, not just the closest one.
- Confidence must be a number between 0 and 1.
- Use the exact existing idea ID and title from the input.
""" + "\n\nINPUT JSON:\n" + json.dumps(
        prompt_data,
        ensure_ascii=False,
    )

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "You are a precise idea-similarity classifier. "
                    "Follow the requested JSON schema exactly."
                )
            }]
        },
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            api_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini text-generation API error {exc.code}: {body}"
        ) from exc

    try:
        text = api_response["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Gemini returned an unexpected response instead of valid JSON."
        ) from exc

    decision = result.get("decision")
    if decision not in {"duplicate", "possible_duplicate", "unique"}:
        raise RuntimeError(
            f"Gemini returned an invalid decision: {decision!r}"
        )

    valid_matches = []
    candidate_by_id = {
        str(item["idea"].get("id", "unknown")): item
        for item in candidates
    }

    for match in result.get("matches", []):
        idea_id = str(match.get("id", ""))

        # Ignore any IDs that were not actually provided to Gemini.
        if idea_id not in candidate_by_id:
            continue

        classification = match.get("classification")
        if classification not in {"duplicate", "possible_duplicate"}:
            continue

        try:
            confidence = float(match.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        valid_matches.append({
            "id": idea_id,
            "title": candidate_by_id[idea_id]["idea"].get(
                "title", "Untitled idea"
            ),
            "classification": classification,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(match.get("reason", "")).strip(),
        })

    # Keep the top-level decision consistent with validated matches.
    if any(
        match["classification"] == "duplicate"
        for match in valid_matches
    ):
        decision = "duplicate"
    elif valid_matches:
        decision = "possible_duplicate"
    else:
        decision = "unique"

    return {
        "decision": decision,
        "matches": valid_matches,
        "summary": str(result.get("summary", "")).strip(),
    }


# ============================================================
# INDEX AND IDEA FILES
# ============================================================

def load_index():
    if not INDEX_FILE.exists():
        return []

    with INDEX_FILE.open("r", encoding="utf-8") as file:
        ideas = json.load(file)

    if not isinstance(ideas, list):
        raise RuntimeError(
            f"{INDEX_FILE} must contain a JSON array of ideas."
        )

    return ideas


def save_index(ideas):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = INDEX_FILE.with_suffix(".json.tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(ideas, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary_file.replace(INDEX_FILE)


def next_idea_id(ideas):
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
    IDEAS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON string quoting is compatible with YAML quoted scalars.
    content = f"""---
id: {idea_id}
issue_number: {issue_number}
username: {json.dumps(username, ensure_ascii=False)}
title: {json.dumps(title, ensure_ascii=False)}
category: {json.dumps(category, ensure_ascii=False)}
submitted_at: {json.dumps(submitted_at)}
keywords: {json.dumps(sorted(keywords), ensure_ascii=False)}
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

    path = IDEAS_DIR / f"idea-{idea_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ============================================================
# GEMINI RESULT FORMATTING
# ============================================================

def format_matches(matches):
    if not matches:
        return "No matching existing ideas were identified."

    lines = []

    for match in matches[:10]:
        lines.append(
            f"### {match['title']} (ID: {match['id']})"
        )
        lines.append(
            f"- Classification: {match['classification']}"
        )
        lines.append(
            f"- Confidence: {match['confidence']:.0%}"
        )

        if match["reason"]:
            lines.append(f"- Reason: {match['reason']}")

        lines.append("")

    if len(matches) > 10:
        lines.append(
            f"_And {len(matches) - 10} additional matches._"
        )

    return "\n".join(lines)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    if not GITHUB_REPOSITORY:
        raise RuntimeError("GITHUB_REPOSITORY is missing.")

    if not ISSUE_NUMBER:
        raise RuntimeError("ISSUE_NUMBER is missing.")

    issue_number = int(ISSUE_NUMBER)

    issue = github_request(
        "GET",
        f"/repos/{GITHUB_REPOSITORY}/issues/{issue_number}",
    )

    username = issue.get("user", {}).get("login") or "unknown"
    title = normalize_issue_title(issue.get("title", ""))
    body = issue.get("body") or ""

    problem = get_field(body, "Problem / Pain Point")
    solution = get_field(body, "Proposed Solution")
    benefit = get_field(body, "Expected Benefit")
    audience = get_field(body, "Who Would Benefit?")
    category = get_field(body, "Category")
    additional = get_field(body, "Additional Details")

    required = {
        "Idea Title": title,
        "Problem / Pain Point": problem,
        "Proposed Solution": solution,
        "Expected Benefit": benefit,
        "Who Would Benefit?": audience,
        "Category": category,
    }

    missing = [
        name for name, value in required.items()
        if not value.strip()
    ]

    if missing:
        add_issue_comment(
            "I couldn't process this idea because these required "
            "fields are missing or empty:\n\n"
            + "\n".join(f"- {name}" for name in missing)
        )
        return

    ideas = load_index()

    new_idea = {
        "title": title,
        "problem": problem,
        "solution": solution,
        "benefit": benefit,
        "audience": audience,
        "category": category,
        "additional": additional,
    }

    new_idea_text = build_idea_text(new_idea)

    print(f"Loaded {len(ideas)} existing ideas.")

    candidates = shortlist_candidates(new_idea_text, ideas)
    print(f"Sending {len(candidates)} candidates to Gemini.")

    result = call_gemini_for_duplicates(new_idea, candidates)

    print(f"Gemini decision: {result['decision']}")
    print(f"Gemini summary: {result['summary']}")

    if result["decision"] == "duplicate":
        add_issue_comment(
            "## Duplicate idea detected\n\n"
            "Gemini found an existing idea that appears to describe "
            "the same underlying problem or solution. This submission "
            "has **not** been added to the index.\n\n"
            + format_matches(result["matches"])
            + "\n\nIf you believe this is meaningfully different, "
            "please explain the distinction in the issue."
        )
        add_issue_labels("duplicate")
        print("Duplicate detected. Submission not recorded.")
        return

    submitted_at = datetime.now(timezone.utc).isoformat()
    idea_id = next_idea_id(ideas)
    keywords = extract_keywords(
        " ".join([
            title, problem, solution, benefit, audience,
            category, additional,
        ])
    )

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

    # Deliberately do not generate or store embeddings.
    new_index_entry = {
        "id": idea_id,
        "issue_number": issue_number,
        "username": username,
        "title": title,
        "category": category,
        "submitted_at": submitted_at,
        "keywords": sorted(keywords),
        "problem": problem,
        "solution": solution,
        "benefit": benefit,
        "audience": audience,
        "description": new_idea_text,
    }

    ideas.append(new_index_entry)
    save_index(ideas)

    if result["decision"] == "possible_duplicate":
        add_issue_comment(
            "## Possible duplicate idea\n\n"
            "Gemini found one or more related ideas. Your submission "
            "has been recorded for human review.\n\n"
            + format_matches(result["matches"])
            + "\n\n"
            "Please review these ideas and clarify any meaningful "
            "differences if needed."
        )
        add_issue_labels("possible-duplicate", "accepted")
    else:
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
