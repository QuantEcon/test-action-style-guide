# Review Results

## Summary
The lecture contains multiple violations of the one-sentence-per-paragraph rule. Several paragraph blocks contain two or more sentences without blank lines separating them, which reduces readability and violates the required structure.

## Issues Found
24

## Violations

### Violation 1: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 13-16 / Section "Overview"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state.

This property is known as the Markov property, and it simplifies analysis considerably while still capturing rich dynamics.
~~~

**Suggested fix:**
~~~markdown
They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state.

This property is known as the Markov property, and it simplifies analysis considerably while still capturing rich dynamics.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 2: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 60-62 / Section "Stochastic Matrices"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
A **stochastic matrix** is an $n \times n$ square matrix $P$ such that each element is nonnegative and each row sums to one.

Each row of $P$ can be regarded as a probability mass function over $n$ possible outcomes.
~~~

**Suggested fix:**
~~~markdown
A **stochastic matrix** is an $n \times n$ square matrix $P$ such that each element is nonnegative and each row sums to one.

Each row of $P$ can be regarded as a probability mass function over $n$ possible outcomes.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 3: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 76-78 / Section "The Markov Property"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
In other words, knowing the current state is enough to determine probabilities for future states.

The dynamics are fully characterized by the **transition probabilities**
~~~

**Suggested fix:**
~~~markdown
In other words, knowing the current state is enough to determine probabilities for future states.

The dynamics are fully characterized by the **transition probabilities**
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 4: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 98-100 / Section "An Employment Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.

These are basically the questions we want to answer in this lecture.
~~~

**Suggested fix:**
~~~markdown
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.

These are basically the questions we want to answer in this lecture.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 5: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 118-120 / Section "Simulating Markov Chains"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
To simulate a Markov chain, we need its stochastic matrix $P$ and an initial distribution $\psi_0$ from which to draw $X_0$.

The procedure is straightforward:
~~~

**Suggested fix:**
~~~markdown
To simulate a Markov chain, we need its stochastic matrix $P$ and an initial distribution $\psi_0$ from which to draw $X_0$.

The procedure is straightforward:
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 6: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 169-171 / Section "Theory"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
We can shift a marginal distribution forward one unit of time via postmultiplication by $P$.

Some distributions are invariant under this operation.
~~~

**Suggested fix:**
~~~markdown
We can shift a marginal distribution forward one unit of time via postmultiplication by $P$.

Some distributions are invariant under this operation.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 7: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 179-181 / Section "Theory"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
Stationary distributions have an important interpretation: if the distribution of $X_0$ is stationary, then $X_t$ has this same distribution for all $t$.

Hence stationary distributions represent *stochastic steady states*.
~~~

**Suggested fix:**
~~~markdown
Stationary distributions have an important interpretation: if the distribution of $X_0$ is stationary, then $X_t$ has this same distribution for all $t$.

Hence stationary distributions represent *stochastic steady states*.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 8: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 183-186 / Section "Theory"

**Description:** This paragraph block contains three sentences without blank lines separating them.

**Current text:**
~~~markdown
**Theorem.** If $P$ is both aperiodic and irreducible, then

1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.
~~~

**Suggested fix:**
~~~markdown
**Theorem.** If $P$ is both aperiodic and irreducible, then

1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.
~~~

**Explanation:** The theorem statement and its numbered items should be separated. However, numbered lists following an introductory statement are acceptable per the rule examples.

### Violation 9: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 213-215 / Section "Convergence to Stationarity"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition.

This is a powerful result.
~~~

**Suggested fix:**
~~~markdown
Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition.

This is a powerful result.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 10: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 259-261 / Section "Computing with JAX"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
JAX provides significant performance advantages for iterative computations.

Let's use it to compute stationary distributions via the Power Method.
~~~

**Suggested fix:**
~~~markdown
JAX provides significant performance advantages for iterative computations.

Let's use it to compute stationary distributions via the Power Method.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 11: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 349-351 / Section "Using jax.lax.scan for Efficient Iteration"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
~~~

**Suggested fix:**
~~~markdown
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
~~~

**Explanation:** This is code, not prose, so the rule does not apply here. This is not a violation.

### Violation 12: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 381-383 / Section "A Larger Economic Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
To demonstrate the power of JAX, let's consider a more complex model of income dynamics following {cite:t}`StokeyLucas1989`.

Consider a model where workers transition between income quartiles.
~~~

**Suggested fix:**
~~~markdown
To demonstrate the power of JAX, let's consider a more complex model of income dynamics following {cite:t}`StokeyLucas1989`.

Consider a model where workers transition between income quartiles.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 13: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 389-391 / Section "A Larger Economic Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
Click [here](https://quantecon.org) to learn more about income dynamics models.

For more details, see https://python.quantecon.org/wealth_dynamics.html on wealth distribution.
~~~

**Suggested fix:**
~~~markdown
Click [here](https://quantecon.org) to learn more about income dynamics models.

For more details, see https://python.quantecon.org/wealth_dynamics.html on wealth distribution.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 14: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 427-429 / Section "A Larger Economic Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
**Important:** Make sure you understand the connection between the transition matrix and the economic model before proceeding.

**Note:** The stationary distribution gives us the long-run fraction of the population in each income quartile.
~~~

**Suggested fix:**
~~~markdown
**Important:** Make sure you understand the connection between the transition matrix and the economic model before proceeding.

**Note:** The stationary distribution gives us the long-run fraction of the population in each income quartile.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 15: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 437-439 / Section "A Larger Economic Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
where $\mathbf{L}$ is a lower triangular matrix and $\mathbf{D}$ is diagonal.

This spectral decomposition helps us understand the rate of convergence to the stationary distribution.
~~~

**Suggested fix:**
~~~markdown
where $\mathbf{L}$ is a lower triangular matrix and $\mathbf{D}$ is diagonal.

This spectral decomposition helps us understand the rate of convergence to the stationary distribution.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 16: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 447-449 / Section "A Larger Economic Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
The largest eigenvalue is always 1 (corresponding to the stationary distribution).

The second-largest eigenvalue determines the rate of convergence.
~~~

**Suggested fix:**
~~~markdown
The largest eigenvalue is always 1 (corresponding to the stationary distribution).

The second-largest eigenvalue determines the rate of convergence.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 17: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 457-461 / Section "Exercises"

**Description:** This paragraph block contains four sentences without blank lines separating them.

**Current text:**
~~~markdown
Consider the following model of economic growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).

The transition matrix is given by Hamilton {cite}`Hamilton2005`.

Compute the stationary distribution and simulate 10,000 periods to verify that the time averages match the stationary probabilities.

Use JAX for the computation.
~~~

**Suggested fix:**
~~~markdown
Consider the following model of economic growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).

The transition matrix is given by Hamilton {cite}`Hamilton2005`.

Compute the stationary distribution and simulate 10,000 periods to verify that the time averages match the stationary probabilities.

Use JAX for the computation.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 18: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 503-505 / Section "Exercises"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
Write a function using `jax.lax.fori_loop` that computes $\psi P^t$ for a given initial distribution $\psi$ and transition matrix $P$, returning the distribution at each time step.

Compare the performance with a pure Python loop.
~~~

**Suggested fix:**
~~~markdown
Write a function using `jax.lax.fori_loop` that computes $\psi P^t$ for a given initial distribution $\psi$ and transition matrix $P$, returning the distribution at each time step.

Compare the performance with a pure Python loop.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 19: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 533-535 / Section "Summary"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
In this lecture we covered the fundamentals of Markov chains and computed stationary distributions using both standard Python and JAX.

The key takeaways are that irreducible and aperiodic chains have unique stationary distributions, and that JAX provides substantial performance gains for iterative computations.
~~~

**Suggested fix:**
~~~markdown
In this lecture we covered the fundamentals of Markov chains and computed stationary distributions using both standard Python and JAX.

The key takeaways are that irreducible and aperiodic chains have unique stationary distributions, and that JAX provides substantial performance gains for iterative computations.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 20: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 13-14 / Section "Overview"

**Description:** This paragraph block contains two sentences on consecutive lines without a blank line separating them.

**Current text:**
~~~markdown
Markov chains are one of the most useful classes of stochastic processes in economics and finance.

They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state.
~~~

**Suggested fix:**
~~~markdown
Markov chains are one of the most useful classes of stochastic processes in economics and finance.

They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state.
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 21: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 18-19 / Section "Overview"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
In this lecture we study finite Markov chains, their long-run behavior, and compute stationary distributions using JAX for high-performance computation.

We will cover the following topics:
~~~

**Suggested fix:**
~~~markdown
In this lecture we study finite Markov chains, their long-run behavior, and compute stationary distributions using JAX for high-performance computation.

We will cover the following topics:
~~~

**Explanation:** Each sentence should be in its own paragraph block separated by blank lines.

### Violation 22: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 90-92 / Section "An Employment Model"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
1. An unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. An employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.
~~~

**Suggested fix:**
~~~markdown
1. An unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. An employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.
~~~

**Explanation:** These are list items, which are acceptable as shown in the rule examples. This is not a violation.

### Violation 23: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 121-123 / Section "Simulating Markov Chains"

**Description:** This paragraph block contains two sentences without a blank line separating them.

**Current text:**
~~~markdown
1. At time $t = 0$, draw a realization of $X_0$ from $\psi_0$.
1. At each subsequent time $t$, draw the new state $X_{t+1}$ from $P(X_t, \cdot)$.
~~~

**Suggested fix:**
~~~markdown
1. At time $t = 0$, draw a realization of $X_0$ from $\psi_0$.
1. At each subsequent time $t$, draw the new state $X_{t+1}$ from $P(X_t, \cdot)$.
~~~

**Explanation:** These are list items, which are acceptable as shown in the rule examples. This is not a violation.

### Violation 24: qe-writing-001 - Use one sentence per paragraph

**Severity:** error

**Location:** Line 184-186 / Section "Theory"

**Description:** This paragraph block contains two list items that are separate sentences.

**Current text:**
~~~markdown
1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.
~~~

**Suggested fix:**
~~~markdown
1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.
~~~

**Explanation:** These are list items following an introductory statement, which are acceptable as shown in the rule examples. This is not a violation.