#!/usr/bin/env python3
"""
Standalone script to test a single rule + prompt against a lecture file.

Usage:
    python scripts/test_rule_prompt.py lectures/markov_chains_jax.md

Edit PROMPT and RULE below to iterate on wording, then re-run to see
how the LLM response changes. The raw LLM response is printed to stdout
and saved to scripts/last_response.md for easy comparison.

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — Edit these to experiment
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 1.0  # Required when using extended thinking
MAX_TOKENS = 64000
USE_EXTENDED_THINKING = True  # Let Claude reason internally before outputting
THINKING_BUDGET = 10000  # Max tokens for internal reasoning

# ---------------------------------------------------------------------------
# PROMPT — This is the base prompt sent to the LLM.
# Edit freely to test different instructions.
# ---------------------------------------------------------------------------

PROMPT = """
You are a style checker for QuantEcon lecture files written in MyST Markdown.

## Task

Find all violations of the provided rule in the lecture document.

First, silently analyze the entire document and identify candidate violations.
Then, verify each candidate — confirm the current text actually violates the rule and the fix changes the text.
Only include confirmed violations in your response. Report 0 if none exist.

## Response Format

```markdown
# Review Results

## Summary
[1-2 sentence summary]

## Issues Found
[NUMBER ONLY]

## Violations

### Violation 1: [rule-code] - [Rule Title]
**Severity:** error
**Location:** Line [X] / Section "[name]"
**Description:** [Why this violates the rule]
**Current text:**
~~~markdown
[exact quote]
~~~
**Suggested fix:**
~~~markdown
[corrected version — MUST be different from current text]
~~~
**Explanation:** [Why this fix resolves the violation]
```

If Issues Found is 0, do not include a Violations section.
""".strip()

# ---------------------------------------------------------------------------
# RULE — This is the rule definition sent to the LLM.
# Edit freely to test different wording, examples, or scope.
# ---------------------------------------------------------------------------

RULE = """
### Rule: qe-writing-001
**Type:** rule
**Title:** Use one sentence per paragraph

**Description:**
Each paragraph block (text separated by blank lines) must contain exactly one sentence. This improves readability and helps readers digest information in clear, focused chunks.

**Important:** A paragraph is defined as text between blank lines in the markdown source. Line breaks within text (without blank lines) do NOT create new paragraphs and punctuation should be examined to determine a sentence. A single sentence may span multiple lines.

**Examples:**

```markdown
<!-- ❌ VIOLATION: Multiple sentences in one paragraph block (NO blank lines) -->
This section introduces the concept of dynamic programming. Dynamic programming is a powerful method for solving optimization problems. We will use it throughout the lecture series.

<!-- ❌ VIOLATION: Multiple sentences even with line breaks (but NO blank lines between) -->
This section introduces the concept of dynamic programming. Dynamic programming
is a powerful method for solving optimization problems. We will use it throughout
the lecture series.

<!-- ✅ CORRECT: Each sentence in its own paragraph block (separated by blank lines) -->
This section introduces dynamic programming.

Dynamic programming is a powerful method for solving optimization problems with recursive structure.

We will use it throughout the lecture series.

<!-- ✅ CORRECT: Single sentence spanning multiple lines (no blank lines within) -->
Dynamic programming is a powerful method for solving optimization problems
with recursive structure.

<!-- ✅ CORRECT: Already following the rule -->
Many economic time series display persistent growth that prevents them from being asymptotically stationary and ergodic.

For example, outputs, prices, and dividends typically display irregular but persistent growth.

Asymptotic stationarity and ergodicity are key assumptions needed to make it possible to learn by applying statistical methods.

<!-- ✅ CORRECT: Lists can be proceeded by an introduction word -->
Here

* $x_t$ is an $n \\times 1$ vector,
* $A$ is an $n \\times n$ stable matrix (all eigenvalues lie within the open unit circle),
* $z_{t+1} \\sim {\\cal N}(0,I)$ is an $m \\times 1$ IID shock,
* $B$ is an $n \\times m$ matrix, and
* $x_0 \\sim {\\cal N}(\\mu_0, \\Sigma_0)$ is a random initial condition for $x$

```

**Key distinction:**
- **Blank line** = Creates new paragraph (required between sentences)
- **Line break** = Does not create new paragraph (allowed WITHIN sentences)
""".strip()


# ---------------------------------------------------------------------------
# Script logic — no need to edit below this line
# ---------------------------------------------------------------------------

def main():
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Get lecture file path
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_rule_prompt.py <lecture_file>")
        print("Example: python scripts/test_rule_prompt.py lectures/markov_chains_jax.md")
        sys.exit(1)

    lecture_path = Path(sys.argv[1])
    if not lecture_path.exists():
        print(f"❌ File not found: {lecture_path}")
        sys.exit(1)

    lecture_content = lecture_path.read_text()

    # Assemble the full prompt (same structure as reviewer.py)
    full_prompt = f"""{PROMPT}

## Style Rule to Check

**IMPORTANT**: Check ONLY for violations of this specific rule. Do not check other rules.

{RULE}

## Lecture to Review

{lecture_content}
"""

    print(f"📋 Model: {MODEL}")
    print(f"🌡️  Temperature: {TEMPERATURE}")
    if USE_EXTENDED_THINKING:
        print(f"🧠 Extended thinking: enabled (budget: {THINKING_BUDGET} tokens)")
    print(f"📄 Lecture: {lecture_path} ({len(lecture_content)} chars)")
    print(f"📏 Rule: {RULE.splitlines()[1].split(':')[1].strip()}")
    print(f"📨 Full prompt: {len(full_prompt)} chars")
    print("⏳ Calling LLM...\n")

    # Call the LLM
    try:
        from anthropic import Anthropic
    except ImportError:
        print("❌ Install anthropic: pip install anthropic")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Build API kwargs
    api_kwargs = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": full_prompt}],
    )
    if USE_EXTENDED_THINKING:
        api_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET,
        }

    try:
        response = client.messages.create(**api_kwargs)
        # With extended thinking, response has thinking + text blocks
        raw_response = ""
        thinking_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                raw_response = block.text
        usage = response.usage
    except Exception as e:
        if "Streaming is required" in str(e) or "10 minutes" in str(e):
            print("📡 Using streaming mode...")
            raw_response = ""
            thinking_text = ""
            stream_kwargs = {k: v for k, v in api_kwargs.items()}
            with client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta':
                            if hasattr(event.delta, 'thinking'):
                                thinking_text += event.delta.thinking
                            elif hasattr(event.delta, 'text'):
                                raw_response += event.delta.text
            usage = None
        else:
            raise

    # Print thinking (if available)
    if thinking_text:
        print("=" * 60)
        print("THINKING (internal reasoning)")
        print("=" * 60)
        print(thinking_text[:2000] + ("..." if len(thinking_text) > 2000 else ""))
        print(f"\n(Total thinking: {len(thinking_text)} chars)")
        print()

    # Print stats
    print("=" * 60)
    print("RAW LLM RESPONSE")
    print("=" * 60)
    print(raw_response)
    print("=" * 60)

    if usage:
        print(f"\n📊 Tokens — input: {usage.input_tokens}, output: {usage.output_tokens}")

    # Count violations vs identical-text issues
    import re
    violation_count = len(re.findall(r"### Violation \d+", raw_response))
    issues_match = re.search(r"## Issues Found\s*\n\s*(\d+)", raw_response)
    issues_found = int(issues_match.group(1)) if issues_match else "?"

    print(f"📋 Issues Found: {issues_found}")
    print(f"📋 Violation blocks: {violation_count}")

    # Check for identical current_text / suggested_fix pairs
    current_texts = re.findall(r"\*\*Current text:\*\*\s*\n~~~markdown\n(.*?)~~~", raw_response, re.DOTALL)
    suggested_fixes = re.findall(r"\*\*Suggested fix:\*\*\s*\n~~~markdown\n(.*?)~~~", raw_response, re.DOTALL)
    identical = sum(1 for c, s in zip(current_texts, suggested_fixes) if c.strip() == s.strip())
    if identical:
        print(f"⚠️  Identical text violations (false positives): {identical}/{violation_count}")
    else:
        print(f"✅ No identical-text false positives detected")

    # Save response
    output_path = Path(__file__).parent / "last_response.md"
    output_path.write_text(raw_response)
    print(f"\n💾 Response saved to: {output_path}")


if __name__ == "__main__":
    main()
