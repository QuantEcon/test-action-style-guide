# test-action-style-guide

Test repository for [action-style-guide](https://github.com/QuantEcon/action-style-guide) — contains test lectures and development scripts for validating the QuantEcon Style Checker.

## Purpose

This repository provides:
1. **Test lectures** with intentional style violations for regression testing
2. **Clean lectures** that should pass all checks (baseline validation)
3. **Development scripts** for prompt iteration and ground truth validation
4. **GitHub workflows** to test the action in a production-like environment

## Test Lectures

| File | Description |
|------|-------------|
| `lectures/markov_chains_jax.md` | Realistic lecture with 42 embedded violations across all 8 categories |
| `lectures/markov_chains_jax.annotated.md` | Catalog of every intentional violation with line numbers and descriptions |
| `lectures/test-lecture-violations.md` | Simpler lecture with intentional violations across all categories |
| `lectures/test-lecture-clean.md` | Clean lecture that should pass all checks |

### markov_chains_jax.md

The primary test lecture. Contains 42 carefully embedded violations:
- **Writing (12):** Multi-sentence paragraphs, wordiness, capitalization, title case, bold/italic, visual elements
- **Math (5):** Transpose notation, matrix brackets, equation formatting, unicode parameters
- **Code (6):** Package installation, Greek letter unicode, timing patterns
- **JAX (5):** Functional programming, NamedTuple, PRNG key management
- **Figures (5):** Line width, titles, captions, spines
- **Links (3):** Link text quality, URL formatting
- **References (2):** Citation syntax
- **Admonitions (4):** Exercise formatting, nesting, directive syntax

See `lectures/markov_chains_jax.annotated.md` for the complete violation catalog.

## Scripts

### `scripts/test_rule_prompt.py`

Standalone script for **prompt iteration testing**. Tests a single rule + prompt against a lecture file without needing the full `qestyle` installation.

```bash
# Requires ANTHROPIC_API_KEY environment variable
python scripts/test_rule_prompt.py lectures/markov_chains_jax.md
```

Features:
- Edit `PROMPT` and `RULE` variables directly in the script to iterate
- Extended thinking toggle (`USE_EXTENDED_THINKING`), configurable budget
- Counts violations and identifies false positives (identical current/suggested text)
- Saves raw LLM response to `scripts/last_response.md` for comparison

### `scripts/find_multisentence.py`

Deterministic **ground truth finder** for `qe-writing-001` (one sentence per paragraph) violations.

```bash
python scripts/find_multisentence.py lectures/markov_chains_jax.md
```

Parses the file structurally (skips code blocks, directives, frontmatter) and reports all paragraph blocks containing multiple sentences. Useful for validating LLM recall against a known baseline.

## Testing with qestyle

Install `qestyle` from the main repo, then run against test lectures:

```bash
# Install qestyle
pip install git+https://github.com/QuantEcon/action-style-guide.git

# Test writing rules
qestyle lectures/markov_chains_jax.md --categories writing

# Dry run (report only, no changes)
qestyle lectures/markov_chains_jax.md --dry-run --categories math

# Reset test files after testing
git checkout -- lectures/
```

## Testing the GitHub Action

### Issue Comment Trigger

Create an issue with a comment like:
```
@qe-style-checker lectures/test-lecture-violations.md math,code
```

### Manual Trigger

The action can be triggered from the Actions tab to test specific categories.

## Related

- [action-style-guide](https://github.com/QuantEcon/action-style-guide) — The main style checker action
- [docs/testing-extended-thinking.md](https://github.com/QuantEcon/action-style-guide/blob/main/docs/testing-extended-thinking.md) — Extended thinking experiment results
