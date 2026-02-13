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
TEMPERATURE = 0.0
MAX_TOKENS = 64000

# ---------------------------------------------------------------------------
# PROMPT — This is the base prompt sent to the LLM.
# Edit freely to test different instructions.
# ---------------------------------------------------------------------------

PROMPT = """
# QuantEcon Writing Style Checker Prompt

You are an expert technical writing editor specializing in QuantEcon lecture materials. Your task is to review a lecture document for violations of **one specific writing style rule** and provide actionable suggestions for improvement.

## Your Role

You will receive:
1. **One specific writing style rule** to check
2. **A lecture document** to review

The rule's `Type:` field indicates how to apply it:
- **`rule`**: Mechanical, objective violations - report all instances found
- **`style`**: Subjective, advisory guidelines - use judgment, only report significant issues

## Instructions

1. **Check ONLY the specific rule provided**: Do not check for other writing issues, even if you notice them. Focus exclusively on the single rule you receive.

2. **Ignore content outside the rule's scope**: Do not check math notation, code blocks, figures, references, or links unless the specific rule applies to them.

3. **Read the entire lecture carefully** to understand its context before identifying violations.

4. **Be thorough and systematic** in checking the specific rule throughout the entire document.

5. **For each violation found**, provide:
   - **Rule Code and Title**: The rule ID and title exactly as provided
   - **Location**: Line number(s) or section heading where the violation occurs
   - **Current Text**: Quote the problematic text exactly as it appears
   - **Issue**: Brief explanation of why this violates the rule
   - **Suggested Fix**: Specific corrected version of the text

6. **Apply the rule appropriately**:
   - **`rule` type**: Report all clear violations mechanically
   - **`style` type**: Use judgment - only report when significantly impacting readability
   - Always explain your reasoning clearly

## Output Format

**CRITICAL**: You MUST structure your response EXACTLY as shown below. The automated parser requires this precise format.

```markdown
# Review Results

## Summary
[Brief 1-2 sentence summary of your findings for this specific rule]

## Issues Found
[JUST THE NUMBER - e.g., 3]

## Violations

### Violation 1: [rule-code] - [Rule Title]

**Severity:** error

**Location:** Line [X-Y] / Section "[Section Name]"

**Description:** [Brief explanation of how this violates the specific rule]

**Current text:**
~~~markdown
[Exact quote of the problematic text - can be multiple lines]
~~~

**Suggested fix:**
~~~markdown
[The corrected version of the text]
~~~

**Explanation:** [Why this change fixes the violation]

[Continue for ALL violations found...]
```

**CRITICAL FORMATTING RULES:**

1. **Issues Found**: Must contain ONLY a number (e.g., `3`, not `3 issues found`)
2. **Violation numbering**: Use sequential numbers (Violation 1, Violation 2, etc.)
3. **Severity levels**: Use `error` for `rule` type, `warning` or `info` for `style` type
4. **Code blocks**: Current text and Suggested fix MUST be in triple-tilde fenced blocks (`~~~markdown`)
5. **Do NOT include** a "Corrected Content" section - fixes will be applied programmatically
6. **Do NOT deviate** from this structure - the parser depends on it
7. **Do NOT report** violations of other rules - only the specific rule provided
8. **NEVER report a violation where the suggested fix is identical to the current text.** If you cannot propose a concrete change that modifies the text, then no violation exists — do NOT report it. A violation means something must change.

**Important**: If NO violations are found for the specific rule, return ONLY this response:

```markdown
# Review Results

## Summary
No violations found for [rule-code]. The lecture follows this rule correctly.

## Issues Found
0
```

**CRITICAL**: When Issues Found is 0, do NOT include a Violations section. Do NOT create violation blocks with "No change needed" or similar commentary as the suggested fix — this causes content to be deleted.
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

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": full_prompt}]
        )
        raw_response = response.content[0].text
        usage = response.usage
    except Exception as e:
        if "Streaming is required" in str(e) or "10 minutes" in str(e):
            print("📡 Using streaming mode...")
            raw_response = ""
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": full_prompt}]
            ) as stream:
                for text in stream.text_stream:
                    raw_response += text
            usage = None
        else:
            raise

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
