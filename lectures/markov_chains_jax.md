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

(mc_jax)=
# Markov Chains and Steady States

## Overview

Markov chains are one of the most useful classes of stochastic processes in economics and finance.

They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state.

This property is known as the **Markov property**, and it simplifies analysis considerably while still capturing rich dynamics.

In this lecture we study finite Markov chains, their long-run behavior, and compute stationary distributions using JAX for high-performance computation.

We will cover the following topics:

- Defining stochastic matrices and the Markov property
- Simulating sample paths
- Computing and interpreting stationary distributions
- Leveraging JAX for fast iterative computation

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

We begin with standard imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
```

## Definitions and setup

### Stochastic matrices

A **stochastic matrix** is an $n \times n$ square matrix $P$ such that each element is nonnegative and each row sums to one.

Each row of $P$ can be regarded as a probability mass function over $n$ possible outcomes.

It is not difficult to show that if $P$ is a stochastic matrix, then so is $P^k$ for all $k \in \mathbb{N}$.

### The Markov property

A **Markov chain** $\{X_t\}$ on a finite state space $S = \{x_1, \ldots, x_n\}$ is a sequence of random variables satisfying

$$
\mathbb{P}\{X_{t+1} = y \mid X_t\} = \mathbb{P}\{X_{t+1} = y \mid X_t, X_{t-1}, \ldots\}
$$

In other words, knowing the current state is enough to determine probabilities for future states.

The dynamics are fully characterized by the **transition probabilities**

$$
P(x, y) := \mathbb{P}\{X_{t+1} = y \mid X_t = x\} \qquad (x, y \in S)
$$

We can view $P$ as a stochastic matrix where $P_{ij} = P(x_i, x_j)$.

## An employment model

Consider a worker who at any given time $t$ is either unemployed (state 0) or employed (state 1).

Suppose that, over a one month period,

1. An unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. An employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.

The transition matrix is

$$
P = \begin{bmatrix}
1 - \alpha & \alpha \\
\beta & 1 - \beta
\end{bmatrix}
$$

Once we have the values of α and β, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.

These are basically the questions we want to answer in this lecture.

Let's set up the model with specific parameter values.

```{code-cell} ipython3
α = 0.3    # probability of finding a job
β = 0.2     # probability of losing a job

P = np.array([[1 - α, α],
              [β, 1 - β]])

print(P)
```

## Simulating Markov chains

To simulate a Markov chain, we need its stochastic matrix $P$ and an initial distribution $\psi_0$ from which to draw $X_0$.

The procedure is straightforward:

1. At time $t = 0$, draw a realization of $X_0$ from $\psi_0$.
1. At each subsequent time $t$, draw the new state $X_{t+1}$ from $P(X_t, \cdot)$.

### Rolling our own

Let's write a simulation function.

```{code-cell} ipython3
def mc_sample_path(P, init=0, sample_size=1_000):
    """Simulate a Markov chain sample path."""
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)
    X[0] = init
    n = len(P)
    # Convert rows to cumulative distributions
    P_dist = [np.cumsum(P[i, :]) for i in range(n)]
    for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])
    return X
```

Let's use our function to simulate the employment model.

```{code-cell} ipython3
X = mc_sample_path(P, sample_size=10_000)
print(f"Fraction of time unemployed: {np.mean(X == 0):.4f}")
```

As we'll see below, the theoretical long-run fraction is $\beta / (\alpha + \beta) = 0.4$.

### Using QuantEcon

The QuantEcon library provides a much faster JIT-compiled implementation.

```{code-cell} ipython3
mc = qe.MarkovChain(P)
X = mc.simulate(ts_length=100_000)

print(f"Fraction unemployed: {np.mean(X == 0):.4f}")
```

```{code-cell} ipython3
import time
start_time = time.time()
X = mc.simulate(ts_length=1_000_000)
end_time = time.time()
print(f"QuantEcon simulation: {end_time - start_time:.4f} seconds")
```

## Stationary distributions

### Theory

We can shift a marginal distribution forward one unit of time via postmultiplication by $P$.

Some distributions are invariant under this operation.

A distribution $\psi^*$ is called **stationary** (or invariant) for $P$ if

$$
\psi^* = \psi^* P
$$ (stationary-dist)

Stationary distributions have an important interpretation: if the distribution of $X_0$ is stationary, then $X_t$ has this same distribution for all $t$.

Hence stationary distributions represent *stochastic steady states*.

**Theorem.** If $P$ is both aperiodic and irreducible, then

1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.

For a proof, see, for example, theorem 5.2 of {cite}`haggstrom2002finite`.

### Computing the stationary distribution

For our employment model, we can find the stationary distribution analytically.

Using $\psi^* = \psi^* P$ and some algebra yields

$$
\psi^* = \left( \frac{\beta}{\alpha + \beta}, \frac{\alpha}{\alpha + \beta} \right)
$$

Let's verify this numerically using QuantEcon.

```{code-cell} ipython3
mc = qe.MarkovChain(P)
psi_star = mc.stationary_distributions[0]
print(f"Stationary distribution: {psi_star}")
print(f"Theoretical: [{beta/(alpha+beta):.4f}, {alpha/(alpha+beta):.4f}]")
```

### Convergence to stationarity

Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition.

This is a powerful result.

The convergence is illustrated in the next figure.

```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

ψ = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("iteration")
ax.set_ylabel("probability")

ψ_history = [ψ.copy()]
for t in range(30):
    ψ = ψ @ P_hamilton
    ψ_history.append(ψ.copy())

ψ_history = np.array(ψ_history)

labels = ["Normal Growth", "Mild Recession", "Severe Recession"]
for i, label in enumerate(labels):
    ax.plot(psi_history[:, i], label=label, lw=2)

mc_h = qe.MarkovChain(P_hamilton)
psi_star_h = mc_h.stationary_distributions[0]
for i in range(3):
    ax.axhline(psi_star_h[i], color='k', linestyle='--', alpha=0.3)

ax.legend()
plt.show()
```

```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: fig-convergence-plot
:width: 80%

Convergence to Hamilton's stationary distribution
```

The figure above shows how the distribution converges over time.

As shown in {numref}`convergence-plot`, the initial condition doesn't matter for the long-run outcome.

## Computing with JAX

JAX provides significant performance advantages for iterative computations.

Let's use it to compute stationary distributions via the Power Method.

```{code-cell} ipython3
import jax.numpy as jnp
from jax import jit
import numpy as np

# Build a larger transition matrix
n = 50
np.random.seed(42)
P_large = np.random.dirichlet(np.ones(n), size=n)
P_large = jnp.array(P_large)
```

### The power method

The power method computes $\psi^*$ by repeatedly applying $\psi \leftarrow \psi P$ until convergence.

```{code-cell} ipython3
from typing import NamedTuple

class StationaryParams(NamedTuple):
    P: jnp.ndarray
    tol: float = 1e-10
    max_iter: int = 10_000
    n: int = 0

def create_stationary_params(P, tol=1e-10, max_iter=10_000):
    """Create stationary distribution solver parameters with validation."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    n = P.shape[0]
    return StationaryParams(P=P, tol=tol, max_iter=max_iter, n=n)

def solve_stationary(params):
    """Solve for stationary distribution using the power method."""
    ψ = jnp.ones(params.n) / params.n
    for i in range(params.max_iter):
        ψ_new = ψ @ params.P
        if jnp.max(jnp.abs(ψ_new - ψ)) < params.tol:
            return ψ_new, i + 1
        ψ = ψ_new
    return ψ, params.max_iter
```

Let's test it.

```{code-cell} ipython3
solver = StationarySolver(P_large)
ψ_star, iterations = solver.solve()
print(f"Converged in {iterations} iterations")
print(f"Distribution sums to: {jnp.sum(ψ_star):.10f}")
```

### JIT-compiled version

We can make this much faster with JAX's JIT compilation.

```{code-cell} ipython3
import jax.numpy as jnp
from jax import jit

@jit
def power_method_jax(P, tol=1e-10, max_iter=10_000):
    n = P.shape[0]
    ψ = jnp.ones(n) / n
    
    def body_fn(carry):
        ψ, i = carry
        ψ_new = ψ @ P
        return ψ_new, i + 1
    
    # Simple iteration
    for i in range(1000):
        ψ = ψ @ P
    
    return ψ
```

```{code-cell} ipython3
import time

start_time = time.time()
result = power_method_jax(P_large)
end_time = time.time()
print(f"JAX power method: {end_time - start_time:.4f} seconds")
```

### Using jax.lax.scan for efficient iteration

For proper JAX-style iteration, we should use `jax.lax.scan` instead of Python loops.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

A much better JAX-idiomatic approach is

```{code-cell} ipython3
@jit
def power_method_scan(P, num_steps=1000):
    n = P.shape[0]
    ψ_0 = jnp.ones(n) / n
    
    def step(ψ, _):
        ψ_new = ψ @ P
        return ψ_new, ψ
    
    ψ_final, ψ_history = jax.lax.scan(step, ψ_0, jnp.arange(num_steps))
    return ψ_final, ψ_history
```

```{code-cell} ipython3
psi_final, history = power_method_scan(P_large)
print(f"Final distribution sums to: {jnp.sum(psi_final):.10f}")
```

## A larger economic model

To demonstrate the power of JAX, let's consider a more complex model of income dynamics following {cite:t}`StokeyLucas1989`.

Consider a model where workers transition between income quartiles.

The model uses a $4 \times 4$ transition matrix.

Click [here](https://quantecon.org) to learn more about income dynamics models.

For more details, see {doc}`intermediate:wealth_dynamics` on wealth distribution.

Also check {doc}`this lecture<intro:markov_chains_I>` for introductory material.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np

# Mixing numpy and jax arrays
income_P = np.array([
    [0.6, 0.3, 0.08, 0.02],
    [0.15, 0.55, 0.25, 0.05],
    [0.05, 0.15, 0.60, 0.20],
    [0.02, 0.08, 0.30, 0.60]
])

# Convert to JAX and compute stationary distribution
income_P_jax = jnp.array(income_P)
ψ = jnp.ones(4) / 4
for _ in range(500):
    ψ = ψ @ income_P_jax

print("Stationary distribution of income quartiles:")
quartile_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
for label, prob in zip(quartile_labels, ψ):
    print(f"  {label}: {prob:.4f}")
```

**Important:** Make sure you understand the connection between the transition matrix and the economic model before proceeding.

**Note:** The stationary distribution gives us the long-run fraction of the population in each income quartile.

The income transition matrix $P$ can be decomposed as

$$
P = L D L^\top
$$

where $L$ is a lower triangular matrix and $D$ is diagonal.

This spectral decomposition helps us understand the rate of convergence to the stationary distribution.

The eigenvalues of $P^\top$ determine how quickly the distribution converges.

```{code-cell} ipython3
eigenvalues = np.linalg.eigvals(income_P.T)
print("Eigenvalues:", np.sort(np.abs(eigenvalues))[::-1])
```

The largest eigenvalue is always 1 (corresponding to the stationary distribution), and the second-largest eigenvalue determines the rate of convergence.

## Exercises

````{prf:exercise}
:label: mc-jax-ex1

Consider the following model of economic growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).

The transition matrix is given by {cite:t}`Hamilton2005`.

Compute the stationary distribution and simulate 10,000 periods to verify that the time averages match the stationary probabilities.

Use JAX for the computation.
````

````{prf:solution} mc-jax-ex1
:class: dropdown
```

```{code-cell} ipython3
import jax.numpy as jnp
import jax.random as jr

P_growth = jnp.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

# Compute stationary distribution
ψ = jnp.ones(3) / 3
for _ in range(1000):
    ψ = ψ @ P_growth

print(f"Stationary distribution: {ψ}")

# Simulate
np.random.seed(42)
key = jr.PRNGKey(42)
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states[0] = 0

P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states[t+1] = np.random.choice(3, p=P_np[states[t]])

for i, label in enumerate(["Recession", "Normal", "Boom"]):
    empirical = np.mean(states == i)
    print(f"  {label}: empirical={empirical:.4f},  theoretical={psi[i]:.4f}")
```

````
```

```{prf:exercise}
:label: mc-jax-ex2

Write a function using `jax.lax.fori_loop` that computes $\psi P^t$ for a given initial distribution $\psi$ and transition matrix $P$, returning the distribution at each time step.

Compare the performance with a pure Python loop.
```

````{prf:solution} mc-jax-ex2
:class: dropdown
```

```{code-cell} ipython3
import jax
import jax.numpy as jnp

@jit
def iterate_distribution(P, psi_0, T):
    def body(t, psi):
        return psi @ P
    return jax.lax.fori_loop(0, T, body, psi_0)

P_test = jnp.array([[0.9, 0.1], [0.3, 0.7]])
psi_0 = jnp.array([1.0, 0.0])

result = iterate_distribution(P_test, psi_0, 100)
print(f"Distribution after 100 steps: {result}")

%timeit iterate_distribution(P_test, psi_0, 1000).block_until_ready()
```

````
```

## Summary

In this lecture we covered the fundamentals of Markov chains and computed stationary distributions using both standard Python and JAX.

The key takeaways are that irreducible and aperiodic chains have unique stationary distributions, and that JAX provides substantial performance gains for iterative computations.

For further reading, see {doc}`intro:markov_chains_I` for the foundations and {doc}`jax:markov_chains_jax` for advanced JAX implementations.

## References

```{bibliography}
```
