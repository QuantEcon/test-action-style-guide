#!/usr/bin/env python3
"""Find all multi-sentence paragraph blocks in a lecture file.

This establishes ground truth for qe-writing-001 violations.
A paragraph block = text between blank lines (not in code/math/directives).
"""

import re
import sys
from pathlib import Path


def find_multisentence_paragraphs(filepath):
    content = Path(filepath).read_text()
    lines = content.split("\n")

    in_code = False
    in_frontmatter = False
    paragraphs = []
    current_lines = []
    start_line = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track YAML frontmatter
        if i == 1 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue

        # Track code blocks and directives (``` or ````)
        if re.match(r"^`{3,}", stripped):
            if in_code:
                in_code = False
                current_lines = []
                continue
            else:
                in_code = True
                if current_lines:
                    paragraphs.append((start_line, i - 1, current_lines))
                    current_lines = []
                continue

        if in_code:
            continue

        # Skip math block delimiters
        if stripped == "$$":
            if current_lines:
                paragraphs.append((start_line, i - 1, current_lines))
                current_lines = []
            continue

        # Skip headers
        if stripped.startswith("#"):
            if current_lines:
                paragraphs.append((start_line, i - 1, current_lines))
                current_lines = []
            continue

        # Blank line = paragraph boundary
        if stripped == "":
            if current_lines:
                paragraphs.append((start_line, i - 1, current_lines))
                current_lines = []
        else:
            if not current_lines:
                start_line = i
            current_lines.append(line)

    if current_lines:
        paragraphs.append((start_line, len(lines), current_lines))

    # Find paragraphs with multiple sentences
    # Heuristic: split on ". " followed by capital letter (or end of common patterns)
    violations = []
    for start, end, para_lines in paragraphs:
        text = " ".join(l.strip() for l in para_lines)

        # Skip list items (start with - or number.)
        if re.match(r"^(\d+\.|[-*])\s", text):
            continue

        # Count sentences: period/!/?  followed by space+capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        if len(sentences) > 1:
            violations.append({
                "start": start,
                "end": end,
                "sentence_count": len(sentences),
                "text": text,
            })

    return violations


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "lectures/markov_chains_jax.md"
    violations = find_multisentence_paragraphs(filepath)

    print(f"Found {len(violations)} multi-sentence paragraph blocks:\n")
    for i, v in enumerate(violations, 1):
        preview = v["text"][:150] + ("..." if len(v["text"]) > 150 else "")
        print(f"  {i}. Line {v['start']}-{v['end']} ({v['sentence_count']} sentences)")
        print(f"     {preview}")
        print()
