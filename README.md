# Idea Hub

A GitHub-based idea submission and semantic duplicate detection system.

## How it works

Users submit ideas through a GitHub Issue Form.

GitHub Actions then:

1. Reads the submitted idea.
2. Builds a semantic representation of the idea.
3. Generates a Gemini embedding.
4. Compares it with existing idea embeddings.
5. Calculates cosine similarity.
6. Uses keyword/Jaccard similarity as a secondary signal.
7. Flags likely duplicate ideas.
8. Stores accepted ideas in the `ideas/` directory.
9. Updates `ideas/index.json`.
10. Adds a comment and label to the GitHub issue.

## Architecture

```text
GitHub Issue Form
       |
       v
GitHub Actions
       |
       v
Python
       |
       v
Gemini Embedding API
       |
       v
Cosine Similarity
       |
       +----------------+
       |                |
       v                v
Similar             New idea
       |                |
       v                v
Review             Store idea
                        |
                        v
                  ideas/index.json
