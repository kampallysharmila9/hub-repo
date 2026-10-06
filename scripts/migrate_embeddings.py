#!/usr/bin/env python3

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    ""
)

INDEX_PATH = Path(
    "ideas/index.json"
)


def fail(message):

    print(
        f"ERROR: {message}",
        file=sys.stderr
    )

    sys.exit(1)


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

        body = (
            error
            .read()
            .decode("utf-8")
        )

        fail(
            f"Gemini API error "
            f"{error.code}: {body}"
        )

    except urllib.error.URLError as error:

        fail(
            f"Gemini connection error: {error}"
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


def read_idea_markdown(file_path):

    if not file_path.exists():
        return ""

    return file_path.read_text(
        encoding="utf-8"
    )


def main():

    if not INDEX_PATH.exists():

        fail(
            "ideas/index.json does not exist."
        )

    try:

        index = json.loads(
            INDEX_PATH.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:

        fail(
            f"Invalid index.json: {error}"
        )

    ideas = index.get(
        "ideas",
        []
    )

    print(
        f"Found {len(ideas)} ideas."
    )

    generated = 0
    skipped = 0

    for idea in ideas:

        idea_id = idea.get(
            "id",
            "unknown"
        )

        existing_embedding = (
            idea.get("embedding")
        )

        if (
            isinstance(
                existing_embedding,
                list
            )
            and len(existing_embedding) > 0
        ):

            print(
                f"Skipping idea #{idea_id} "
                f"- embedding already exists."
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Try to read the original Markdown file.
        # ----------------------------------------------------

        file_name = idea.get(
            "file"
        )

        text = ""

        if file_name:

            text = read_idea_markdown(
                Path(file_name)
            )

        # ----------------------------------------------------
        # Fallback to index fields if Markdown isn't available.
        # ----------------------------------------------------

        if not text:

            text = f"""
Idea Title:
{idea.get("title", "")}

Problem:
{idea.get("problem", "")}

Proposed Solution:
{idea.get("solution", idea.get("idea_text", ""))}

Expected Benefit:
{idea.get("benefit", "")}

Category:
{idea.get("category", "")}

Keywords:
{", ".join(idea.get("keywords", []))}
""".strip()

        print("")
        print(
            f"Generating embedding "
            f"for idea #{idea_id}..."
        )

        embedding = generate_embedding(
            text
        )

        idea["embedding"] = embedding

        generated += 1

        # Save after every idea.
        # If the job stops, running it again will
        # continue with the remaining ideas.

        INDEX_PATH.write_text(
            json.dumps(
                index,
                indent=2,
                ensure_ascii=False
            ) + "\n",
            encoding="utf-8"
        )

        print(
            f"✓ Idea #{idea_id} completed."
        )

    print("")
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(
        f"Embeddings generated: {generated}"
    )
    print(
        f"Already had embeddings: {skipped}"
    )


if __name__ == "__main__":
    main()
