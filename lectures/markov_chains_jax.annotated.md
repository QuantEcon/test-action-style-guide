# Annotated Violations: markov_chains_jax.md

This document catalogs every intentional style violation embedded in
`markov_chains_jax.md`. Use it to measure precision/recall of `qestyle`
and to detect regressions when prompts or rules change.

**Total violations:** 42
**Across categories:** writing (12), math (5), code (6), jax (5), figures (5),
links (3), references (2), admonitions (4)

---

## Writing violations (12)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| W1 | qe-writing-001 | 16 | Two sentences in one paragraph | "Markov chains are one of the most useful classes...finance. They provide a framework..." |
| W2 | qe-writing-001 | 18 | Two sentences in one paragraph | "...the Markov property, and it simplifies...dynamics." (sentence continues but next sentence starts mid-paragraph) |
| W3 | qe-writing-001 | 82 | Two sentences in one paragraph | "Once we have the values of α and β, we can address questions like...long run. These are basically the questions we want to answer in this lecture." |
| W4 | qe-writing-002 | 16 | Wordy opening sentence | "Markov chains are one of the most useful classes of stochastic processes in economics and finance." — overly long introduction could be tighter |
| W5 | qe-writing-004 | 57 | Unnecessary capitalization | "Current State" — mid-sentence capitalization of common noun |
| W6 | qe-writing-004 | 78 | Unnecessary capitalization | "The Transition Matrix is" — capitalize "Transition Matrix" |
| W7 | qe-writing-004 | 131 | Unnecessary capitalization | "Stochastic Steady States" — mid-sentence capitalization |
| W8 | qe-writing-004 | 148 | Unnecessary capitalization | "Stationary Distribution" — mid-sentence in "we can find the Stationary Distribution analytically" |
| W9 | qe-writing-004 | 206 | Unnecessary capitalization | "Rate of Convergence" — mid-sentence capitalization |
| W10 | qe-writing-005 | 193–194 | Bold used for emphasis, not definition | `**Important:**` and `**Note:**` use bold for emphasis labels |
| W11 | qe-writing-006 | 219 | Section heading over-capitalized | "## A Larger Economic Model" — "Larger" and "Economic" and "Model" shouldn't be capitalized in a section heading |
| W12 | qe-writing-008 | 82 | Extra whitespace | "questions like what is the average duration of unemployment,  and what fraction" — double space before "and" |

## Math violations (5)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| M1 | qe-math-003 | 80 | Uses pmatrix instead of bmatrix | `\begin{pmatrix}` for the transition matrix |
| M2 | qe-math-004 | 196–198 | Bold face for matrices/vectors | `\mathbf{P}`, `\mathbf{L}`, `\mathbf{D}`, `\mathbf{L}^T` |
| M3 | qe-math-002 | 198 | Uses ^T instead of ^\top for transpose | `\mathbf{L}^T` |
| M4 | qe-math-007 | 127 | Manual equation tag | `\tag{1}` in the stationary distribution equation |
| M5 | qe-math-001 | 82 | LaTeX in narrative text | Uses `$\alpha$` and `$\beta$` for isolated parameter mentions in a sentence with no math expressions (could use α and β) |

## Code violations (6)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| C1 | qe-code-002 | 88–89 | Spelled-out Greek letters in code | `alpha = 0.3` and `beta = 0.2` instead of `α` and `β` |
| C2 | qe-code-003 | 30 | pip install missing hide-output tag | `!pip install quantecon` code cell has no `tags: [hide-output]` |
| C3 | qe-code-004 | 117–120 | Manual time.time() pattern | Uses `start_time = time.time()` / `end_time - start_time` instead of `qe.Timer()` |
| C4 | qe-code-004 | 179–182 | Manual time.time() pattern (2nd occurrence) | Same manual timing pattern in JAX section |
| C5 | qe-code-005 | 255 | Jupyter magic for benchmarking | `%timeit` instead of `qe.timeit()` |
| C6 | qe-fig-003 | 159 | Matplotlib embedded title | `ax.set_title("Convergence to Stationary Distribution")` |

## JAX violations (5)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| J1 | qe-jax-002 | 170–178 | Class instead of NamedTuple | `class StationarySolver` with mutable `self.` attributes |
| J2 | qe-jax-005 | 188–190 | Python loop in JIT-able function | `for i in range(1000): psi = psi @ P` inside JIT function |
| J3 | qe-jax-001 | 192–196 | Imperative pattern, not pure function | `update_distribution` uses a Python list accumulator `result = []` and `result.append(...)` |
| J4 | qe-jax-006 | 229–231 | NumPy random instead of JAX PRNG | `np.random.seed(42)` and `np.random.choice(3, ...)` in JAX code, even though `jr.PRNGKey` is imported but not used for simulation |
| J5 | qe-jax-004 | — | Mixing numpy and jax arrays | `income_P = np.array(...)` then `income_P_jax = jnp.array(income_P)` — should create directly as `jnp.array` |

## Figures violations (5)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| F1 | qe-fig-003 | 159 | Matplotlib embedded title | `ax.set_title(...)` (also counted under code) |
| F2 | qe-fig-004 | 172 | Caption too long and over-capitalized | "Convergence of Marginal Distributions to the Stationary Distribution for Hamilton's Recession Model" — 11 words, Title Case |
| F3 | qe-fig-006 | 160–161 | Uppercase axis labels | `ax.set_xlabel("Iteration")` and `ax.set_ylabel("Probability")` — should be lowercase |
| F4 | qe-fig-008 | 165 | Line width not set to 2 | `lw=2` is present on some lines but the call pattern is only in the loop — other plot calls in the lecture (line ~109) don't specify `lw` |
| F5 | qe-fig-001 | 155 | Unnecessary figsize setting | `figsize=(10, 6)` explicitly set when defaults from `_config.yml` should apply |

## Links violations (3)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| L1 | qe-link-001 | 222 | Bare URL with poor link text | `Click [here](https://quantecon.org) to learn more` — "here" is generic link text |
| L2 | qe-link-001 | 224 | Bare URL for same-series content | `https://python.quantecon.org/wealth_dynamics.html` — should use `{doc}` link |
| L3 | qe-link-002 | 226 | Full URL instead of doc link | `https://intro.quantecon.org/markov_chains_I.html` — should be `{doc}\`intro:markov_chains_I\`` |

## References violations (2)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| R1 | qe-ref-001 | 137 | Wrong citation style | `{cite}\`haggstrom2002finite\`` — author name is part of sentence flow, so should use `{cite:t}` |
| R2 | qe-ref-001 | 212 | Citation style correct (control) | `{cite:t}\`StokeyLucas1989\`` — this is CORRECT and should NOT be flagged (false positive test) |

## Admonitions violations (4)

| # | Rule | Line(s) | Description | Embedded text |
|---|------|---------|-------------|---------------|
| A1 | qe-admon-002 | — | Solution without dropdown class | `{solution-start} mc-jax-ex2` does include `:class: dropdown` (control — correct) |
| A2 | qe-writing-005 | 193–194 | Should be admonitions, not bold text | `**Important:**` and `**Note:**` should use `{important}` and `{note}` admonition directives |
| A3 | qe-admon-001 | 217–240 | Code cells inside non-gated exercise | `{exercise}` block on ex1 uses the non-gated ```` syntax but the solution contains code cells |
| A4 | qe-admon-003 | 217 | Tick count management | The exercise directive uses 4 backticks but could cause nesting issues with code blocks inside |

---

## Correct patterns (should NOT be flagged)

These items are intentionally correct and serve as false-positive tests:

| # | Rule | What's correct |
|---|------|----------------|
| FP1 | qe-writing-001 | Most paragraphs in the lecture follow one-sentence-per-paragraph |
| FP2 | qe-writing-005 | **stochastic matrix**, **Markov chain**, **Markov property**, **transition probabilities**, **stationary** — all bold definitions |
| FP3 | qe-math-006 | No `\begin{align}` misuse (all display math uses `$$`) |
| FP4 | qe-ref-001 | `{cite:t}\`StokeyLucas1989\`` used correctly for in-text citation |
| FP5 | qe-ref-001 | `{cite}\`Hamilton2005\`` used correctly at end of sentence |
| FP6 | qe-jax-005 | `jax.lax.scan` used correctly in `power_method_scan` |
| FP7 | qe-jax-005 | `jax.lax.fori_loop` used correctly in exercise 2 solution |
| FP8 | qe-admon-002 | Both solutions use `:class: dropdown` |
| FP9 | qe-link-002 | `{doc}\`intro:markov_chains_I\`` and `{doc}\`jax:markov_chains_jax\`` used correctly in Summary |
| FP10 | qe-code-001 | PEP8 followed in most code cells |

---

## Using this document

### Measuring precision and recall

After running `qestyle markov_chains_jax.md`, compare the report against this list:

- **True positive (TP):** violation listed here AND flagged by qestyle
- **False negative (FN):** violation listed here but NOT flagged
- **False positive (FP):** flagged by qestyle but NOT listed here (and not a legitimate issue)
- **True negative (TN):** correct pattern (FP1–FP10) NOT flagged

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

### Tracking regressions

Run qestyle before and after prompt/rule changes. Compare:
1. Total violations detected
2. Which specific violations from this list were found/missed
3. Whether any false positives from the FP table appeared

### Category-specific testing

```bash
# Test individual categories
qestyle markov_chains_jax.md --dry-run -c writing
qestyle markov_chains_jax.md --dry-run -c math
qestyle markov_chains_jax.md --dry-run -c code
qestyle markov_chains_jax.md --dry-run -c jax
qestyle markov_chains_jax.md --dry-run -c figures
qestyle markov_chains_jax.md --dry-run -c links
qestyle markov_chains_jax.md --dry-run -c references
qestyle markov_chains_jax.md --dry-run -c admonitions
```
