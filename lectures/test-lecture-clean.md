---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Test Lecture: Clean Example

This lecture demonstrates proper style that should pass all checks in the QuantEcon Style Checker.

## Overview

This lecture covers fundamental concepts in computational economics. The reader will learn
about numerical methods and their applications to economic models.

## Mathematical Foundations

We begin with the key equation for our analysis.

```{math}
:label: eq-bellman
V(x) = \max_{a \in A} \left\{ u(x, a) + \beta \sum_{x'} p(x' | x, a) V(x') \right\}
```

Equation {eq}`eq-bellman` is the Bellman equation, which forms the foundation of dynamic programming.
The discount factor $\beta \in (0, 1)$ determines how much the agent values future rewards relative
to current rewards.

For a simple linear model, we have $y = \alpha + \beta x + \varepsilon$, where $\varepsilon$ is
the error term with $E[\varepsilon] = 0$.

## Code Implementation

The following code implements our numerical solution.

```{code-cell} ipython3
:tags: [hide-output]

import numpy as np
import matplotlib.pyplot as plt

# Define model parameters
alpha = 0.5
beta = 0.95
n_states = 100

# Create state space
x = np.linspace(0, 10, n_states)

# Define utility function
def utility(x, a):
    """Compute utility from state x and action a."""
    return np.log(x + 1) - 0.5 * a**2
```

Now we can visualize the results.

```{code-cell} ipython3
:tags: []

# Generate sample data
y = utility(x, alpha * x)

# Create visualization
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label='Utility', linewidth=2)
ax.set_xlabel('State $x$')
ax.set_ylabel('Utility $u(x, a)$')
ax.set_title('Utility Function')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Key Results

Our analysis reveals several important findings.

```{prf:theorem}
:label: thm-convergence

Under assumptions A1-A3, the value function iteration converges to the unique fixed point
$V^*$ at a geometric rate with factor $\beta$.
```

See {prf:ref}`thm-convergence` for the formal statement of convergence.

```{note}
The convergence rate depends on the discount factor. Higher values of $\beta$ lead to
slower convergence but more accurate long-run predictions.
```

## Practical Considerations

When implementing these methods, practitioners should consider computational efficiency.
The algorithm complexity scales with the size of the state space, making careful
discretization essential for large problems.

```{tip}
Start with a coarse grid to verify the code works correctly, then refine the
discretization for production runs.
```

## Summary

This lecture introduced the Bellman equation and demonstrated a basic implementation.
The key takeaways are:

1. Dynamic programming provides a systematic approach to sequential decision problems
2. Value function iteration converges under standard conditions
3. Computational efficiency requires careful attention to discretization

## References

```{bibliography}
:filter: docname in docnames
```
