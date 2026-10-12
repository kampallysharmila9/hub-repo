#!/usr/bin/env python3
"""Bulk duplicate detection using Gemini embeddings + cosine similarity only."""
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from embedding_utils import INDEX_PATH, cosine_similarity, generate_embedding, idea_text, load_index, save_index

DUPLICATE_THRESHOLD = float(os.environ.get("DUPLICATE_THRESHOLD", "0.82"))
POSSIBLE_DUPLICATE_THRESHOLD = float(os.environ.get("POSSIBLE_DUPLICATE_THRESHOLD", "0.70"))
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
REPORT_PATH = Path("duplicate_reports/duplicates.json")


def github_request(method, endpoint, payload=None):
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("Warning: GitHub credentials/repository missing; skipping issue labels/comments.", file=sys.stderr)
        return None
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/{endpoint}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gemini-embedding-duplicate-detector",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        detail = exc.read().decode("utf-8", errors="replace") if isinstance(exc, urllib.error.HTTPError) else str(exc)
        print(f"Warning: GitHub API request failed ({endpoint}): {detail}", file=sys.stderr)
        return None


def add_label(issue_number, label):
    if issue_number:
        github_request("POST", f"issues/{issue_number}/labels", {"labels": [label]})


def add_comment(issue_number, message):
    if issue_number:
        github_request("POST", f"issues/{issue_number}/comments", {"body": message})


def main():
    ideas, shape = load_index()
    print("=" * 60)
    print("GEMINI EMBEDDING DUPLICATE DETECTION")
    print("=" * 60)
    print(f"Found {len(ideas)} ideas.")
    if len(ideas) < 2:
        print("Not enough ideas to compare.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps({"total_ideas": len(ideas), "duplicate_clusters": []}, indent=2) + "\n", encoding="utf-8")
        return

    # Generate missing vectors first, then persist them. Existing vectors are retained.
    generated = 0
    for idea in ideas:
        emb = idea.get("embedding")
        if not isinstance(emb, list) or not emb:
            text = idea_text(idea)
            if not text:
                print(f"Warning: idea #{idea.get('id', '?')} has no text; skipping.", file=sys.stderr)
                continue
            print(f"Generating missing embedding for idea #{idea.get('id', '?')}...")
            idea["embedding"] = generate_embedding(text)
            generated += 1
        elif len(emb) != int(os.environ.get("GEMINI_EMBEDDING_DIMENSIONS", "768")):
            raise RuntimeError(
                f"Idea #{idea.get('id', '?')} has an embedding of dimension {len(emb)}; "
                "all vectors must use the configured Gemini model and dimensions. Run the migration script."
            )
    if generated:
        save_index(ideas, shape)
    print(f"Embeddings generated: {generated}")

    valid = [idea for idea in ideas if isinstance(idea.get("embedding"), list) and idea.get("embedding")]
    clusters = []
    edges = []
    adjacency = {str(i.get("id", idx)): set() for idx, i in enumerate(valid)}
    id_map = {str(i.get("id", idx)): i for idx, i in enumerate(valid)}
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            score = cosine_similarity(a["embedding"], b["embedding"])
            if score >= POSSIBLE_DUPLICATE_THRESHOLD:
                classification = "duplicate" if score >= DUPLICATE_THRESHOLD else "possible_duplicate"
                edge = {
                    "idea_a_id": a.get("id"), "idea_a_title": a.get("title", "Untitled idea"),
                    "idea_a_issue_number": a.get("issue_number"),
                    "idea_b_id": b.get("id"), "idea_b_title": b.get("title", "Untitled idea"),
                    "idea_b_issue_number": b.get("issue_number"),
                    "similarity": round(score, 4), "classification": classification,
                }
                edges.append(edge)
                aid, bid = str(a.get("id", i)), str(b.get("id", j))
                adjacency.setdefault(aid, set()).add(bid)
                adjacency.setdefault(bid, set()).add(aid)
                print(f"{classification}: #{a.get('id')} <-> #{b.get('id')} (cosine={score:.3f})")

    # Connected components form clusters; only report components with >=2 ideas.
    visited = set()
    for node in adjacency:
        if node in visited or not adjacency[node]:
            continue
        stack, component = [node], []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency.get(current, set()) - visited)
        if len(component) > 1:
            members = [id_map[x] for x in component if x in id_map]
            clusters.append({
                "idea_ids": [item.get("id") for item in members],
                "ideas": [{"id": item.get("id"), "title": item.get("title"), "issue_number": item.get("issue_number")} for item in members],
            })

    # Label/comment secondary ideas in each cluster; this is a review signal, not auto-closure.
    for cluster in clusters:
        members = cluster["ideas"]
        primary = members[0]
        for member in members[1:]:
            issue_number = member.get("issue_number")
            add_label(issue_number, "possible-duplicate")
            add_comment(issue_number, (
                "## Possible duplicate detected (Gemini embeddings)\n\n"
                f"This idea is semantically similar to **{primary.get('title', 'Untitled idea')}** "
                f"(Idea ID: {primary.get('id')}, Issue: #{primary.get('issue_number')}).\n\n"
                "Cosine similarity is a screening signal; please review both ideas before deciding whether they are duplicates."
            ))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "Gemini embeddings + cosine similarity",
        "embedding_model": os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview"),
        "embedding_dimensions": int(os.environ.get("GEMINI_EMBEDDING_DIMENSIONS", "768")),
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "possible_duplicate_threshold": POSSIBLE_DUPLICATE_THRESHOLD,
        "total_ideas": len(ideas),
        "ideas_with_embeddings": len(valid),
        "generated_embeddings": generated,
        "matches": sorted(edges, key=lambda item: item["similarity"], reverse=True),
        "duplicate_clusters": clusters,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=" * 60)
    print(f"Duplicate-like pairs: {len(edges)}")
    print(f"Clusters: {len(clusters)}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
