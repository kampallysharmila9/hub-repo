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
        fail("GEMINI_API_KEY is missing.")

    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{EMBEDDING_MODEL}:embedContent"
    )

    payload = {
        "model": f"models/{EMBEDDING_MODEL}",

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

        body = error.read().decode("utf-8")

        fail(
            f"Gemini API error "
            f"{error.code}: {body}"
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

    for idea in ideas:

        idea_id = idea.get(
            "id",
            "unknown"
        )

        if (
            isinstance(
                idea.get("embedding"),
                list
            )
            and idea.get("embedding")
        ):

            print(
                f"Skipping idea #{idea_id} "
                f"- embedding already exists."
            )

            continue

        file_name = idea.get(
            "file"
        )

        text = ""

        if file_name:

            path = Path(file_name)

            if path.exists():

                text = path.read_text(
                    encoding="utf-8"
                )

        if not text:

            text = f"""
Idea Title:
{idea.get("title", "")}

Problem / Pain Point:
{idea.get("problem", "")}

Proposed Solution:
{idea.get("solution", "")}

Expected Benefit:
{idea.get("benefit", "")}

Category:
{idea.get("category", "")}

Keywords:
{", ".join(idea.get("keywords", []))}
""".strip()

        print(
            f"Generating embedding "
            f"for idea #{idea_id}..."
        )

        idea["embedding"] = generate_embedding(
            text
        )

        generated += 1

        INDEX_PATH.write_text(
            json.dumps(
                index,
                indent=2,
                ensure_ascii=False
            ) + "\n",
            encoding="utf-8"
        )

        print(
            f"Completed idea #{idea_id}"
        )

    print("")
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(
        f"Embeddings generated: {generated}"
    )


if __name__ == "__main__":
    main()
