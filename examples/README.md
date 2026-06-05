# Pre-executed `qestyle` examples

Captured output from running `qestyle` on the flagship test lecture
[`lectures/markov_chains_jax.md`](../lectures/markov_chains_jax.md) — a realistic
QuantEcon lecture seeded with **42 catalogued style violations**.

These are committed so the results can be **shown without a live run** (no API
key, no wait) — useful for demos and as a regression snapshot.

| File | What it is |
|------|------------|
| [`qestyle-all-report.md`](qestyle-all-report.md) | Full review report — 38 style suggestions + 47 applied fixes, by rule |
| [`markov_chains_jax.fixes.diff`](markov_chains_jax.fixes.diff) | The exact changes `qestyle` applied to the lecture (`git diff`) |
| [`recall-analysis.md`](recall-analysis.md) | Precision/recall vs the 42-violation catalog — **27/30 rules (90%)** |

## Snapshot

- **Version:** qestyle v0.7.2 · **Date:** 2026-06-05 · all 8 categories
- **86 issues** — 47 auto-applied fixes, 38 human-review suggestions, 3 warnings
- **Rule-level recall:** 27 / 30 seeded rules (90%)

> Output is **not deterministic** (temperature 1.0 + extended thinking) — a fresh
> run will differ slightly. See [`recall-analysis.md`](recall-analysis.md) for detail.

## Regenerate

```bash
qestyle lectures/markov_chains_jax.md            # all categories, applies fixes
git diff lectures/markov_chains_jax.md > examples/markov_chains_jax.fixes.diff
cp "lectures/qestyle(all)-markov_chains_jax.md" examples/qestyle-all-report.md
git checkout -- lectures/markov_chains_jax.md    # reset the lecture
rm "lectures/qestyle(all)-markov_chains_jax.md"
```
