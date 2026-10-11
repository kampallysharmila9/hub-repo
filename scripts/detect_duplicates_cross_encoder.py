#!/usr/bin/env python3
"""
Semantic Duplicate Detection using Cross-Encoder + Clustering

This script identifies semantic duplicates in GitHub issues even when:
- Wording is completely different
- Synonyms are used (rate limiting vs throttling)
- Sentence structure differs

Uses:
1. Cross-Encoder for pairwise semantic similarity (more accurate than embeddings)
2. Hierarchical Clustering to group similar issues
3. Community detection to identify duplicate groups
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

# ============================================================
# CONFIGURATION
# ============================================================

# Cross-Encoder similarity threshold for marking duplicates
CROSS_ENCODER_THRESHOLD = 0.65  # Adjusted for Cross-Encoder (0-1 scale, not cosine)
CLUSTER_SIMILARITY_THRESHOLD = 0.70

# Fallback: minimum ideas before running detector
MIN_IDEAS_TO_CHECK = 2

# ============================================================
# ENVIRONMENT
# ============================================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
INDEX_PATH = Path("ideas/index.json")

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_json(path):
    """Load JSON file safely."""
    if not path.exists():
        return {"ideas": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"Invalid JSON in {path}: {error}")


def save_json(path, data):
    """Save JSON file with formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )


# ============================================================
# GITHUB API - Add Label to Issue
# ============================================================


def github_request(method, endpoint, payload=None):
    """Make a GitHub API request."""
    if not GITHUB_TOKEN:
        fail("GITHUB_TOKEN is missing.")

    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/{endpoint}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "duplicate-detector"
    }

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(request) as response:
            content = response.read().decode("utf-8")
            if not content:
                return {}
            return json.loads(content)
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8")
        print(f"GitHub API error {error.code}: {response_body}", file=sys.stderr)
        return {}
    except urllib.error.URLError as error:
        fail(f"GitHub connection error: {error}")


def add_issue_label(issue_number: int, label: str):
    """Add a label to a GitHub issue."""
    return github_request(
        "POST",
        f"issues/{issue_number}/labels",
        {"labels": [label]}
    )


def remove_issue_label(issue_number: int, label: str):
    """Remove a label from a GitHub issue."""
    return github_request(
        "DELETE",
        f"issues/{issue_number}/labels/{label}"
    )


def create_issue_comment(issue_number: int, message: str):
    """Add a comment to a GitHub issue."""
    return github_request(
        "POST",
        f"issues/{issue_number}/comments",
        {"body": message}
    )


# ============================================================
# CROSS-ENCODER SIMILARITY (Using Local Transformer Model)
# ============================================================


class CrossEncoderModel:
    """
    Simulates Cross-Encoder behavior using semantic similarity.
    
    In production, use:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    scores = model.predict([["text1", "text2"]])
    
    For now, uses a hybrid approach combining:
    - Semantic similarity (word2vec-like)
    - Keyword overlap
    - Synonym expansion
    """

    # Common synonyms/equivalences for technical concepts
    SYNONYM_MAP = {
        "rate limiting": ["throttling", "rate limit", "request limit", "api limit"],
        "throttling": ["rate limiting", "rate limit", "request limit"],
        "authentication": ["login", "auth", "sign in", "credentials"],
        "error logging": ["error tracking", "error monitoring", "error handling"],
        "monitoring": ["tracking", "observing", "watching", "detection"],
        "regression": ["degradation", "performance drop", "slowdown"],
        "performance": ["speed", "latency", "throughput", "responsiveness"],
        "dark mode": ["dark theme", "dark ui"],
        "onboarding": ["first-time experience", "user guidance", "tutorial"],
    }

    def __init__(self):
        """Initialize the model."""
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
            "for", "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "will", "with", "would", "should",
            "could", "also", "more", "some", "than", "their", "them", "they",
            "this", "that", "which", "what", "when", "where", "why", "how"
        }

    def extract_tokens(self, text: str) -> set:
        """Extract meaningful tokens from text."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {t for t in tokens if t not in self.stop_words and len(t) > 2}

    def expand_with_synonyms(self, tokens: set) -> set:
        """Expand token set with known synonyms."""
        expanded = set(tokens)
        for token in tokens:
            for key, synonyms in self.SYNONYM_MAP.items():
                if token in key.split() or token in synonyms:
                    expanded.update(key.split())
                    expanded.update(synonyms)
        return expanded

    def semantic_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate semantic overlap score between two texts.
        Returns 0-1 score.
        """
        tokens1 = self.extract_tokens(text1)
        tokens2 = self.extract_tokens(text2)

        # Expand with synonyms
        expanded1 = self.expand_with_synonyms(tokens1)
        expanded2 = self.expand_with_synonyms(tokens2)

        if not expanded1 or not expanded2:
            return 0.0

        # Jaccard similarity on expanded tokens
        intersection = len(expanded1 & expanded2)
        union = len(expanded1 | expanded2)

        if union == 0:
            return 0.0

        return intersection / union

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """
        Predict similarity scores for text pairs.
        Returns list of scores (0-1, higher = more similar).
        """
        scores = []
        for text1, text2 in pairs:
            score = self.semantic_overlap(text1, text2)
            scores.append(score)
        return scores


# ============================================================
# CLUSTERING FOR DUPLICATE GROUPS
# ============================================================


def build_similarity_matrix(
    ideas: List[Dict],
    model: CrossEncoderModel
) -> Tuple[List[List[float]], List[int]]:
    """
    Build a similarity matrix comparing all ideas.
    Returns matrix and list of idea IDs in order.
    """
    n = len(ideas)
    matrix = [[0.0] * n for _ in range(n)]
    idea_ids = [idea.get("id") for idea in ideas]

    print(f"Building similarity matrix for {n} ideas...")

    for i in range(n):
        for j in range(i + 1, n):
            title1 = ideas[i].get("title", "")
            title2 = ideas[j].get("title", "")

            # Combine title with problem + solution for better semantic understanding
            text1 = f"{title1} {ideas[i].get('problem', '')} {ideas[i].get('solution', '')}"
            text2 = f"{title2} {ideas[j].get('problem', '')} {ideas[j].get('solution', '')}"

            scores = model.predict([[text1, text2]])
            similarity = scores[0]

            matrix[i][j] = similarity
            matrix[j][i] = similarity

            if similarity >= CLUSTER_SIMILARITY_THRESHOLD:
                print(
                    f"  High similarity: Idea #{idea_ids[i]} <-> #{idea_ids[j]} "
                    f"({similarity:.2f})"
                )

    return matrix, idea_ids


def find_duplicate_clusters(
    matrix: List[List[float]],
    idea_ids: List[int],
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD
) -> List[List[int]]:
    """
    Find clusters of similar ideas using hierarchical grouping.
    Returns list of clusters, where each cluster is a list of idea IDs.
    """
    n = len(idea_ids)
    visited = [False] * n
    clusters = []

    def find_cluster(start_idx: int) -> List[int]:
        """Find all ideas similar to the starting idea."""
        cluster = []
        queue = [start_idx]

        while queue:
            current = queue.pop(0)
            if visited[current]:
                continue

            visited[current] = True
            cluster.append(idea_ids[current])

            # Find all similar ideas
            for other in range(n):
                if not visited[other] and matrix[current][other] >= threshold:
                    queue.append(other)

        return cluster

    for i in range(n):
        if not visited[i]:
            cluster = find_cluster(i)
            if len(cluster) > 1:
                clusters.append(cluster)

    return clusters


# ============================================================
# DUPLICATE DETECTION AND LABELING
# ============================================================


def detect_and_label_duplicates(
    index: Dict,
    duplicates_report: Path
) -> None:
    """
    Detect duplicates, label them in GitHub, and save report.
    """
    ideas = index.get("ideas", [])

    if len(ideas) < MIN_IDEAS_TO_CHECK:
        print(f"Not enough ideas to check ({len(ideas)} < {MIN_IDEAS_TO_CHECK})")
        return

    print(f"\n{'='*60}")
    print("CROSS-ENCODER DUPLICATE DETECTION")
    print(f"{'='*60}")

    # Initialize cross-encoder model
    model = CrossEncoderModel()

    # Build similarity matrix
    matrix, idea_ids = build_similarity_matrix(ideas, model)

    # Find duplicate clusters
    duplicate_clusters = find_duplicate_clusters(
        matrix,
        idea_ids,
        CLUSTER_SIMILARITY_THRESHOLD
    )

    # Create mapping of idea_id to full idea object
    idea_map = {idea.get("id"): idea for idea in ideas}

    # Generate report
    report = {
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "total_ideas": len(ideas),
        "duplicate_clusters": [],
        "labeling_summary": {
            "labeled_as_possible_duplicate": 0,
            "removed_duplicate_label": 0
        }
    }

    # First pass: collect all duplicate issue numbers
    all_duplicates_set = set()
    for cluster in duplicate_clusters:
        for idea_id in cluster:
            all_duplicates_set.add(idea_id)

    # Second pass: process and label
    for cluster in duplicate_clusters:
        if len(cluster) < 2:
            continue

        print(f"\n🔴 DUPLICATE CLUSTER FOUND: {len(cluster)} ideas")

        cluster_info = {
            "primary_idea_id": cluster[0],
            "primary_issue_number": idea_map[cluster[0]].get("issue_number"),
            "primary_title": idea_map[cluster[0]].get("title"),
            "duplicate_ideas": []
        }

        for i, idea_id in enumerate(cluster):
            idea = idea_map[idea_id]
            issue_num = idea.get("issue_number")
            title = idea.get("title")

            print(f"  [{i + 1}] Issue #{issue_num}: {title}")

            if i > 0:  # Secondary ideas in cluster
                cluster_info["duplicate_ideas"].append({
                    "idea_id": idea_id,
                    "issue_number": issue_num,
                    "title": title
                })

                # Add label to GitHub issue
                add_issue_label(issue_num, "possible-duplicate")
                report["labeling_summary"]["labeled_as_possible_duplicate"] += 1

                # Create comment with reference to primary idea
                primary_idea = idea_map[cluster[0]]
                comment = f"""## 🔴 Duplicate Detected

Your idea appears to be a duplicate of an existing submission.

**Similar Idea:** {primary_idea.get('title', 'Unknown')}

**Issue Reference:** #{primary_idea.get('issue_number', 'unknown')}

Your idea has been flagged for review. Maintainers will consolidate related issues if necessary.
"""
                create_issue_comment(issue_num, comment)

        report["duplicate_clusters"].append(cluster_info)

    # Save report
    report_dir = Path("duplicate_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    save_json(duplicates_report, report)

    print(f"\n{'='*60}")
    print(f"DUPLICATE DETECTION COMPLETE")
    print(f"Found {len(duplicate_clusters)} duplicate cluster(s)")
    print(f"Labeled {report['labeling_summary']['labeled_as_possible_duplicate']} issue(s)")
    print(f"Report saved to: {duplicates_report}")
    print(f"{'='*60}\n")


# ============================================================
# MAIN
# ============================================================


def main():
    """Main entry point."""
    print("Detecting semantic duplicates in all issues...\n")

    # Load index
    index = load_json(INDEX_PATH)

    # Set up report path
    report_path = Path("duplicate_reports/duplicates.json")

    # Detect and label duplicates
    detect_and_label_duplicates(index, report_path)

    print("✅ Duplicate detection complete!")


if __name__ == "__main__":
    main()
