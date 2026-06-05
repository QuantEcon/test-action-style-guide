# Precision / recall snapshot — qestyle v0.7.2

**Run:** `qestyle lectures/markov_chains_jax.md` (all 8 categories)
**Date:** 2026-06-05 · **Version:** qestyle v0.7.2 · **Model:** claude-sonnet-4-5 (extended thinking)
**Ground truth:** [`markov_chains_jax.annotated.md`](../lectures/markov_chains_jax.annotated.md) — 42 seeded violations across 30 distinct rules.

> Generated output is **not deterministic** (temperature 1.0 + extended thinking).
> Numbers will shift run to run; this is a representative snapshot.

## High-level results

| Metric | Count |
|--------|-------|
| Total issues reported | **86** |
| Applied fixes (auto, rule-type) | **47** |
| Style suggestions (human review) | **38** |
| Warnings (apply/empty issues) | **3** |

## Rule-level recall: 27 / 30 (90%)

`qestyle` flagged at least one violation for **27 of the 30 seeded rules**. This is
up from **26/30 (87%)** at v0.7.0 — `qe-admon-001` is now detected.

### 3 rules missed

| Rule | Violation | Note |
|------|-----------|------|
| `qe-writing-008` | W12 — double space before "and" (line ~82) | Possibly normalised by an earlier writing fix; not surfaced |
| `qe-fig-008` | F4 — line width not set to `lw=2` | **Persistent miss** (also missed at v0.7.0) |
| `qe-link-001` | L1, L2 — bare URL / generic "here" link text | **Persistent miss** (also missed at v0.7.0) — `qe-link-002` fixes may transform the URLs first |

`qe-fig-008` and `qe-link-001` are missed across **both** versions — the two
rules most worth a prompt-engineering pass.

## Additional rules fired (beyond the seeded 30)

Seven rules fired that aren't in the catalog. These are **either genuine extra
finds or false positives — they need a human spot-check**:

`qe-writing-003`, `qe-writing-007`, `qe-math-009`, `qe-jax-003`, `qe-jax-007`,
`qe-fig-002`, `qe-fig-005`

Two are already flagged as low-quality by the run's own warnings:
- `qe-writing-007` — "current text and suggested fix are identical" (no-op / likely FP)
- `qe-fig-002` — "missing suggested_fix" (incomplete)

## False-positive controls

The catalog seeds deliberate *correct* patterns that should **not** be flagged
(FP1–FP10). One to verify by hand from [`qestyle-all-report.md`](qestyle-all-report.md):

- `qe-ref-001` fired once (entry #42, a `Hamilton` in-text citation). Confirm it
  matches catalog **R1**-style redundancy and is **not** flagging control **FP5**
  (`{cite}` used correctly) or **R2** (`{cite:t}` StokeyLucas1989, correct).

## What this snapshot does and doesn't claim

- **Does:** rule-level recall (which rules fired) is measured against the catalog — 90%.
- **Doesn't:** instance-level precision is **not** fully scored here. That requires
  reading each of the 86 entries against the catalog — which is exactly the kind of
  validation the annotated ground-truth file makes possible, and a good live exercise.

## Reproduce

```bash
# from the test-action-style-guide repo root, with qestyle installed + ANTHROPIC_API_KEY set
qestyle lectures/markov_chains_jax.md          # all categories, applies fixes
git diff lectures/markov_chains_jax.md         # inspect the applied changes
git checkout -- lectures/markov_chains_jax.md   # reset
```
