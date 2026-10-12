#!/usr/bin/env python3
"""Generate Gemini embeddings for existing ideas that do not have one yet."""
from embedding_utils import INDEX_PATH, generate_embedding, idea_text, load_index, save_index


def main():
    ideas, shape = load_index()
    print(f"Found {len(ideas)} ideas.")
    generated = 0
    for idea in ideas:
        existing = idea.get("embedding")
        if isinstance(existing, list) and existing:
            print(f"Skipping idea #{idea.get('id', '?')} - embedding already exists.")
            continue
        text = idea_text(idea)
        if not text:
            print(f"Skipping idea #{idea.get('id', '?')} - no text fields or Markdown file found.")
            continue
        print(f"Generating embedding for idea #{idea.get('id', '?')}...")
        idea["embedding"] = generate_embedding(text)
        generated += 1
        save_index(ideas, shape)
        print(f"Completed idea #{idea.get('id', '?')}")
    print("\nMIGRATION COMPLETE")
    print(f"Embeddings generated: {generated}")
    print(f"Index updated: {INDEX_PATH}")


if __name__ == "__main__":
    main()
