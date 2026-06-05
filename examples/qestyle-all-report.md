# Style Guide Report: markov_chains_jax.md

- **Date:** 2026-06-05 16:32
- **Version:** qestyle v0.7.2
- **Issues found:** 86
- **Mode:** fix (rule violations applied to file)

## 📝 Style Suggestions (38)

> **Action required:** These suggestions require human review and judgment.

### 1. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 71 / Section "An employment model"
**Description:** Sentence is 34 words long and contains unnecessary phrasing ("Once we have the values of") that reduces clarity. The compound question structure makes it harder to parse.

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.
```

**Suggestion:**

```
With $\alpha$ and $\beta$, we can answer questions like: What is the average unemployment duration? What fraction of time does a worker spend unemployed long-term?
```

**Explanation:** Removes unnecessary words ("the values of", "address"), shortens the main clause, and breaks the complex question into clearer parts. Reduces from 34 to 28 words while improving clarity.

### 2. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 73 / Section "An employment model"
**Description:** Contains filler word "basically" and verbose phrasing "we want to answer". The sentence is also somewhat redundant with the previous sentence.

```
These are basically the questions we want to answer in this lecture.
```

**Suggestion:**

```
We address these questions in this lecture.
```

**Explanation:** Removes the filler word "basically" and simplifies "we want to answer" to "address", making the sentence more direct and concise while preserving the meaning.

### 3. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 365 / Section "A larger economic model"
**Description:** Verbose phrasing "Make sure you understand" adds unnecessary words. The imperative "Understand" is more direct and equally clear.

```
*Important:* Make sure you understand the connection between the transition matrix and the economic model before proceeding.
```

**Suggestion:**

```
*Important:* Understand the connection between the transition matrix and the economic model before proceeding.
```

**Explanation:** Removes "Make sure you" which adds three words without adding value. The direct imperative is clearer and more concise.

### 4. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 302 / Section "Convergence to stationarity"
**Description:** Sentence is redundant and adds minimal value since the figure already has a detailed caption ({numref}`convergence-plot`) that describes what is shown. The next sentence provides the actual insight.

```
The figure above shows how the distribution converges over time.
```

**Suggestion:**

```
As shown in {numref}`convergence-plot`, the initial condition doesn't matter for the long-run outcome.
```

**Explanation:** Removes the redundant sentence entirely. The following sentence already references the figure and provides substantive insight, eliminating the need for a purely descriptive sentence that duplicates the figure caption.

### 5. qe-writing-003 — Maintain logical flow
**Location:** Line 347-353 / Section "A larger economic model"
**Description:** Multiple external hyperlinks interrupt the narrative flow and distract readers by sending them to other resources instead of maintaining focus on the current lecture's content.

```
Click [here](https://quantecon.org) to learn more about income dynamics models.

For more details, see https://python.quantecon.org/wealth_dynamics.html on wealth distribution.

Also check [this lecture](https://intro.quantecon.org/markov_chains_I.html) for introductory material.
```

**Suggestion:**

```
This income dynamics model is related to wealth distribution models discussed elsewhere in the QuantEcon lectures.
```

**Explanation:** Consolidates the references into a single, non-distracting sentence that maintains focus on the current lecture while acknowledging related material exists.

### 6. qe-writing-003 — Maintain logical flow
**Location:** Line 372-385 / Section "A larger economic model"
**Description:** Introduces spectral decomposition and eigenvalue theory abruptly without establishing prerequisites or explaining why this mathematical machinery is relevant. This creates a jarring jump from practical computation to advanced linear algebra.

```
The income transition matrix $\mathbf{P}$ can be decomposed as

$$
\mathbf{P} = \mathbf{L} \mathbf{D} \mathbf{L}^T
$$

where $\mathbf{L}$ is a lower triangular matrix and $\mathbf{D}$ is diagonal.

This spectral decomposition helps us understand the rate of convergence to the stationary distribution.

The eigenvalues of $P^T$ determine how quickly the distribution converges.
```

**Suggestion:**

```
We can gain insight into the convergence speed by examining the transition matrix's eigenvalues. The second-largest eigenvalue (in absolute value) determines how quickly the distribution converges to stationarity, with values closer to zero indicating faster convergence.
```

**Explanation:** Removes the unmotivated spectral decomposition formula and focuses on the practical insight about eigenvalues and convergence speed, which connects better to the stationary distribution discussion earlier in the lecture.

### 7. qe-writing-003 — Maintain logical flow
**Location:** Line 295-302 / Section "Using jax.lax.scan for efficient iteration"
**Description:** Introduces a function that is never used and immediately dismissed as inferior, creating a distraction from the main narrative.

```
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

**Suggestion:**

```
The JAX-idiomatic approach uses `jax.lax.scan` for efficient iteration:
```

**Explanation:** Removes the tangential "bad example" function and directly introduces the recommended approach, maintaining clearer focus on the correct implementation pattern.

### 8. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 85-104 / Section "An employment model"
**Description:** The employment model describes a two-state Markov chain with transitions between employment states, but presents only the mathematical transition matrix without a visual state diagram. This is a classic visual concept that would benefit from illustration.

```
Consider a worker who at any given time $t$ is either unemployed (state 0) or employed (state 1).

Suppose that, over a one month period,

1. An unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. An employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.

The transition matrix is

$$
P = \begin{pmatrix}
1 - \alpha & \alpha \\
\beta & 1 - \beta
\end{pmatrix}
$$
```

**Suggestion:**

```
Consider a worker who at any given time $t$ is either unemployed (state 0) or employed (state 1).

Suppose that, over a one month period,

1. An unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. An employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.
```

**Explanation:** Adding a state transition diagram would visually show the two states (employed/unemployed) and the transition probabilities between them, making the model structure immediately clear to readers.

### 9. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 135-142 / Section "Simulating Markov chains"
**Description:** The simulation code produces sample paths but only displays numerical summary statistics without visualizing the actual time series, missing an opportunity to show what a Markov chain sample path looks like.

```
Let's use our function to simulate the employment model.
```

**Suggestion:**

```
Let's use our function to simulate the employment model.
```

**Explanation:** Plotting the sample path provides visual intuition for how the Markov chain evolves over time, showing the random transitions between states that readers have just learned to simulate.

### 10. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 391-395 / Section "A larger economic model"
**Description:** Important notes are formatted as plain italic text rather than using admonition blocks, which are specifically designed to draw attention to important information.

```
*Important:* Make sure you understand the connection between the transition matrix and the economic model before proceeding.

*Note:* The stationary distribution gives us the long-run fraction of the population in each income quartile.
```

**Suggestion:**

```
```{note}
Make sure you understand the connection between the transition matrix and the economic model before proceeding.
```

**Explanation:** MyST admonitions provide visual styling (colored boxes, icons) that make important notes stand out from regular text, enhancing readability and ensuring key points aren't missed.

### 11. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 376-391 / Section "A larger economic model"
**Description:** The income quartile model produces a stationary distribution over four income classes but only prints text output. A bar chart would make the distribution and economic interpretation more intuitive.

```
print("Stationary distribution of income quartiles:")
quartile_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
for label, prob in zip(quartile_labels, psi):
    print(f"  {label}: {prob:.4f}")
```

**Suggestion:**

```
print("Stationary distribution of income quartiles:")
quartile_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
for label, prob in zip(quartile_labels, psi):
    print(f"  {label}: {prob:.4f}")

# Visualize the stationary distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(quartile_labels, psi, color='steelblue', alpha=0.7)
ax.set_ylabel('Probability')
ax.set_title('Stationary Distribution Across Income Quartiles')
ax.set_ylim(0, max(psi) * 1.1)
for i, (label, prob) in enumerate(zip(quartile_labels, psi)):
    ax.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
```

**Explanation:** A bar chart immediately shows the relative probabilities across income quartiles, making it easier to see which income levels are most common in the long run and understand the economic implications of the transition dynamics.

### 12. qe-math-009 — Choose simplicity in mathematical notation
**Location:** Line 53 / Section "Definitions and setup"
**Description:** Uses blackboard bold $\mathbb{N}$ for natural numbers when a simpler description would be clearer and sufficient for this applied context.

```
It is not difficult to show that if $P$ is a stochastic matrix, then so is $P^k$ for all $k \in \mathbb{N}$.
```

**Suggestion:**

```
It is not difficult to show that if $P$ is a stochastic matrix, then so is $P^k$ for all positive integers $k$.
```

**Explanation:** Replacing $\mathbb{N}$ with "positive integers" removes unnecessary decorative notation while maintaining clarity. This is more accessible and aligns with the applied nature of QuantEcon lectures.

### 13. qe-math-009 — Choose simplicity in mathematical notation
**Location:** Line 59 / Section "The Markov property"
**Description:** Uses blackboard bold $\mathbb{P}$ for probability when simpler notation would suffice in this applied context.

```
$$
\mathbb{P}\{X_{t+1} = y \mid X_t\} = \mathbb{P}\{X_{t+1} = y \mid X_t, X_{t-1}, \ldots\}
$$
```

**Suggestion:**

```
$$
P(X_{t+1} = y \mid X_t) = P(X_{t+1} = y \mid X_t, X_{t-1}, \ldots)
$$
```

**Explanation:** Using $P(\cdot)$ instead of $\mathbb{P}\{\cdot\}$ simplifies the notation without loss of clarity. While the document uses $P$ for the transition matrix elsewhere, context makes clear when $P$ refers to probability vs. the matrix.

### 14. qe-math-009 — Choose simplicity in mathematical notation
**Location:** Line 67 / Section "The Markov property"
**Description:** Uses blackboard bold $\mathbb{P}$ unnecessarily when defining transition probabilities that are already denoted with the simpler $P(x,y)$.

```
$$
P(x, y) := \mathbb{P}\{X_{t+1} = y \mid X_t = x\} \qquad (x, y \in S)
$$
```

**Suggestion:**

```
$$
P(x, y) := P(X_{t+1} = y \mid X_t = x) \qquad (x, y \in S)
$$
```

**Explanation:** Since the simpler notation $P(x,y)$ is being defined for the transition probability, there's no need to use the more formal $\mathbb{P}$ notation on the right side. Using consistent $P$ notation throughout maintains simplicity.

### 15. qe-code-001 — Follow PEP8 unless closer to mathematical notation
**Location:** Line 119 (within `mc_sample_path` function) / Section "Rolling our own"
**Description:** Binary operator `+` in array indexing lacks required spaces. PEP8 requires spaces around binary operators for readability.

```
```python
    for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])
    return X
```

**Suggestion:**

```
```python
    for t in range(sample_size - 1):
        X[t + 1] = qe.random.draw(P_dist[X[t]])
    return X
```

**Explanation:** PEP8 requires spaces around binary operators including `+`. Array indexing expressions should follow the same spacing rules unless there is mathematical justification, which does not apply to simple index arithmetic.

### 16. qe-code-001 — Follow PEP8 unless closer to mathematical notation
**Location:** Line 165 / Section "Computing the stationary distribution"
**Description:** Binary operators `/` and `+` lack required spaces in f-string expressions. These are computational expressions, not mathematical notation display, so PEP8 spacing applies.

```
```python
print(f"Theoretical: [{β/(α+β):.4f}, {α/(α+β):.4f}]")
```

**Suggestion:**

```
```python
print(f"Theoretical: [{β / (α + β):.4f}, {α / (α + β):.4f}]")
```

**Explanation:** While Greek letters are acceptable for variable names to match mathematical notation, operator spacing should still follow PEP8 conventions for code readability. The expressions are computations, not rendered mathematical formulas.

### 17. qe-code-001 — Follow PEP8 unless closer to mathematical notation
**Location:** Line 451 (within exercise solution) / Section "Exercises"
**Description:** Binary operator `+` in array indexing lacks required spaces.

```
```python
P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states[t+1] = np.random.choice(3, p=P_np[states[t]])
```

**Suggestion:**

```
```python
P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states[t + 1] = np.random.choice(3, p=P_np[states[t]])
```

**Explanation:** Array index arithmetic should include spaces around binary operators per PEP8 standards, improving code readability and consistency.

### 18. qe-code-004 — Use quantecon Timer context manager
**Location:** Line 172 / Section "Using QuantEcon"
**Description:** Manual timing pattern using `time.time()` with start/end time variables and print statement. This should use the `qe.Timer()` context manager instead.

```
```{code-cell} ipython3
import time
start_time = time.time()
X = mc.simulate(ts_length=1_000_000)
end_time = time.time()
print(f"QuantEcon simulation: {end_time - start_time:.4f} seconds")
```

**Suggestion:**

```
```{code-cell} ipython3
print("QuantEcon simulation:")
with qe.Timer():
    X = mc.simulate(ts_length=1_000_000)
```

**Explanation:** The `qe.Timer()` context manager automatically measures and prints elapsed time, eliminating the need for manual `time.time()` calls and making the code more concise and maintainable.

### 19. qe-code-004 — Use quantecon Timer context manager
**Location:** Line 291 / Section "JIT-compiled version"
**Description:** Manual timing pattern using `time.time()` to measure JAX power method performance. This verbose pattern should be replaced with `qe.Timer()` context manager.

```
```{code-cell} ipython3
import time

start_time = time.time()
result = power_method_jax(P_large)
end_time = time.time()
print(f"JAX power method: {end_time - start_time:.4f} seconds")
```

**Suggestion:**

```
```{code-cell} ipython3
print("JAX power method:")
with qe.Timer():
    result = power_method_jax(P_large)
```

**Explanation:** Using `qe.Timer()` provides consistent timing output across the lecture and removes boilerplate code for manual time tracking, adhering to the modern QuantEcon timing API.

### 20. qe-code-005 — Use quantecon timeit for benchmarking
**Location:** Line 136-142 / Section "Using QuantEcon"
**Description:** Manual benchmarking using `time.time()` with start/end times instead of `qe.timeit()` for statistical timing analysis.

```
```{code-cell} ipython3
import time
start_time = time.time()
X = mc.simulate(ts_length=1_000_000)
end_time = time.time()
print(f"QuantEcon simulation: {end_time - start_time:.4f} seconds")
```

**Suggestion:**

```
```{code-cell} ipython3
result = qe.timeit(lambda: mc.simulate(ts_length=1_000_000), number=10)
print(f"QuantEcon simulation: {result.average:.4f} seconds (average over {result.loops} runs)")
```

**Explanation:** Replaces manual timing with `qe.timeit()`, which provides statistical analysis across multiple runs. Uses a lambda function to pass the simulation call with arguments.

### 21. qe-code-005 — Use quantecon timeit for benchmarking
**Location:** Line 312-316 / Section "JIT-compiled version"
**Description:** Manual benchmarking using `time.time()` for measuring JAX function performance instead of `qe.timeit()`.

```
```{code-cell} ipython3
import time

start_time = time.time()
result = power_method_jax(P_large)
end_time = time.time()
print(f"JAX power method: {end_time - start_time:.4f} seconds")
```

**Suggestion:**

```
```{code-cell} ipython3
timing_result = qe.timeit(lambda: power_method_jax(P_large), number=100)
print(f"JAX power method: {timing_result.average:.4f} seconds (average over {timing_result.loops} runs)")
result = power_method_jax(P_large)
```

**Explanation:** Uses `qe.timeit()` for proper statistical benchmarking across multiple runs. Separately assigns the actual result for use in subsequent code.

### 22. qe-code-005 — Use quantecon timeit for benchmarking
**Location:** Line 595 / Section "Exercise mc-jax-ex2 solution"
**Description:** Uses Jupyter magic command `%timeit` instead of `qe.timeit()` for benchmarking.

```
%timeit iterate_distribution(P_test, psi_0, 1000).block_until_ready()
```

**Suggestion:**

```
timing = qe.timeit(lambda: iterate_distribution(P_test, psi_0, 1000).block_until_ready(), number=100)
print(f"Average time: {timing.average:.6f} seconds")
```

**Explanation:** Replaces the Jupyter magic command with `qe.timeit()` for consistent, reproducible benchmarking that works in all environments.

### 23. qe-jax-001 — Use functional programming patterns
**Location:** Line 361 / Section "Using jax.lax.scan for efficient iteration"
**Description:** The function `update_distribution` uses a mutable Python list with `.append()` method calls, which is a side effect. This violates JAX's functional programming principles that encourage pure functions without mutation or side effects.

```
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

**Suggestion:**

```
def update_distribution(psi, P):
    """Update the distribution by one step."""
    return P.T @ psi
```

**Explanation:** The corrected version uses JAX's native matrix operations (`P.T @ psi`) instead of manually building a result through in-place list mutations. This is a pure function with no side effects, adhering to JAX's functional programming paradigm. The matrix-vector product is also more efficient and idiomatic.

### 24. qe-jax-003 — Use generate_path for sequence generation
**Location:** Line 350-360 / Section "Using jax.lax.scan for efficient iteration"
**Description:** The `power_method_scan` function implements a custom `jax.lax.scan` pattern for iterative sequence generation that duplicates the functionality of the standardized `generate_path` pattern. This function generates a path of distribution states over multiple iterations.

```
```{code-cell} ipython3
@jit
def power_method_scan(P, num_steps=1000):
    n = P.shape[0]
    psi_0 = jnp.ones(n) / n
    
    def step(psi, _):
        psi_new = psi @ P
        return psi_new, psi
    
    psi_final, psi_history = jax.lax.scan(step, psi_0, jnp.arange(num_steps))
    return psi_final, psi_history
```

**Suggestion:**

```
```{code-cell} ipython3
from functools import partial

@partial(jax.jit, static_argnames=['num_steps'])
def generate_path(f, initial_state, num_steps, **kwargs):
    """Generate a path of states using iterative function application."""
    def update_wrapper(state, t):
        next_state = f(state, t, **kwargs)
        return next_state, state
    _, path = jax.lax.scan(update_wrapper, initial_state, jnp.arange(num_steps))
    return path.T

def update_distribution(psi, t, P):
    """Update distribution by one step."""
    return psi @ P

@jit
def power_method_scan(P, num_steps=1000):
    n = P.shape[0]
    psi_0 = jnp.ones(n) / n
    psi_history = generate_path(update_distribution, psi_0, num_steps, P=P)
    psi_final = psi_history[:, -1]
    return psi_final, psi_history
```

**Explanation:** This fix refactors the code to use the standardized `generate_path` pattern. The update logic is extracted into a separate `update_distribution` function that takes the state, time index, and transition matrix as parameters. The `generate_path` function handles the scanning logic in a reusable way, eliminating the custom scan implementation and following the established QuantEcon JAX pattern for sequence generation.

### 25. qe-jax-005 — Use jax.lax for control flow
**Location:** Line 384 / Section "Computing with JAX"
**Description:** The `solve_stationary` function uses a Python for loop with JAX arrays (`jnp`) for iterative computation. This prevents proper JIT optimization and should use `jax.lax.while_loop` for conditional iteration or `jax.lax.fori_loop` for fixed iterations.

```
```python
def solve_stationary(params: StationaryParams):
    """Solve for stationary distribution using the power method."""
    n = params.P.shape[0]
    psi = jnp.ones(n) / n
    for i in range(params.max_iter):
        psi_new = psi @ params.P
        if jnp.max(jnp.abs(psi_new - psi)) < params.tol:
            return psi_new, i + 1
        psi = psi_new
    return psi, params.max_iter
```

**Suggestion:**

```
```python
def solve_stationary(params: StationaryParams):
    """Solve for stationary distribution using the power method."""
    n = params.P.shape[0]
    psi = jnp.ones(n) / n
    
    def cond_fn(state):
        psi, psi_new, i = state
        converged = jnp.max(jnp.abs(psi_new - psi)) < params.tol
        return (i < params.max_iter) & (~converged)
    
    def body_fn(state):
        _, psi, i = state
        psi_new = psi @ params.P
        return psi, psi_new, i + 1
    
    psi_init = psi @ params.P
    _, psi_final, iterations = jax.lax.while_loop(
        cond_fn, body_fn, (psi, psi_init, 1)
    )
    return psi_final, iterations
```

**Explanation:** This fix uses `jax.lax.while_loop` for the conditional iteration pattern, which enables proper JIT compilation and tracing of the control flow.

### 26. qe-jax-005 — Use jax.lax for control flow
**Location:** Line 422 / Section "JIT-compiled version"
**Description:** The `power_method_jax` function is decorated with `@jit` but contains a Python for loop, which defeats the purpose of JIT compilation and should be replaced with `jax.lax.fori_loop`.

```
```python
@jit
def power_method_jax(P, tol=1e-10, max_iter=10_000):
    n = P.shape[0]
    psi = jnp.ones(n) / n
    
    def body_fn(carry):
        psi, i = carry
        psi_new = psi @ P
        return psi_new, i + 1
    
    # Simple iteration
    for i in range(1000):
        psi = psi @ P
    
    return psi
```

**Suggestion:**

```
```python
@jit
def power_method_jax(P, tol=1e-10, max_iter=10_000):
    n = P.shape[0]
    psi = jnp.ones(n) / n
    
    def body_fn(i, psi):
        return psi @ P
    
    # Use fori_loop for fixed iterations
    psi = jax.lax.fori_loop(0, 1000, body_fn, psi)
    
    return psi
```

**Explanation:** This fix replaces the Python for loop with `jax.lax.fori_loop`, enabling proper JIT compilation and better performance.

### 27. qe-jax-005 — Use jax.lax for control flow
**Location:** Line 450 / Section "Using jax.lax.scan for efficient iteration"
**Description:** The `update_distribution` function uses a Python for loop with JAX operations. This imperative pattern should be replaced with vectorized operations or JAX control flow.

```
```python
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

**Suggestion:**

```
```python
def update_distribution(psi, P):
    """Update the distribution by one step."""
    return psi @ P
```

**Explanation:** This operation is a simple matrix multiplication and doesn't require any loop at all. The vectorized form is both simpler and enables JAX optimizations.

### 28. qe-jax-005 — Use jax.lax for control flow
**Location:** Line 501 / Section "A larger economic model"
**Description:** Python for loop used for iterative computation with JAX arrays. This should use `jax.lax.fori_loop` for better performance and JIT compatibility.

```
```python
# Convert to JAX and compute stationary distribution
income_P_jax = jnp.array(income_P)
psi = jnp.ones(4) / 4
for _ in range(500):
    psi = psi @ income_P_jax
```

**Suggestion:**

```
```python
# Convert to JAX and compute stationary distribution
income_P_jax = jnp.array(income_P)
psi = jnp.ones(4) / 4

def step(i, psi):
    return psi @ income_P_jax

psi = jax.lax.fori_loop(0, 500, step, psi)
```

**Explanation:** Using `jax.lax.fori_loop` enables JIT compilation of the iteration and better performance, which is especially important for demonstrating JAX advantages in this section.

### 29. qe-jax-005 — Use jax.lax for control flow
**Location:** Line 543 / Section "Exercises" (Solution to mc-jax-ex1)
**Description:** Python for loop used with JAX arrays in the stationary distribution computation. Should use `jax.lax.fori_loop` for consistency with JAX best practices.

```
```python
# Compute stationary distribution
psi = jnp.ones(3) / 3
for _ in range(1000):
    psi = psi @ P_growth
```

**Suggestion:**

```
```python
# Compute stationary distribution
psi = jnp.ones(3) / 3

def step(i, psi):
    return psi @ P_growth

psi = jax.lax.fori_loop(0, 1000, step, psi)
```

**Explanation:** Using `jax.lax.fori_loop` demonstrates proper JAX control flow patterns, which is especially important in exercise solutions that students will learn from.

### 30. qe-jax-007 — Use consistent function naming for updates
**Location:** Line ~395 / Section "Using jax.lax.scan for efficient iteration"
**Description:** The function `update_distribution` violates the naming convention by placing "update" before the quantity name instead of after it. Additionally, it's missing the `time_step` parameter that should be included for consistency even if unused.

```
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

**Suggestion:**

```
def distribution_update(psi, time_step, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

**Explanation:** The fix renames the function from `update_distribution` to `distribution_update` to follow the `[quantity]_update` pattern specified in the rule. It also adds the `time_step` parameter between the current state (`psi`) and the model/parameters (`P`) for consistency with the documented pattern, even though this parameter is not used in the function body.

### 31. qe-jax-004 — Use functional update patterns
**Location:** Line 115 / Section "Rolling our own"
**Description:** In-place array assignment `X[0] = init` violates JAX functional update pattern. Even in NumPy code, JAX lectures should demonstrate functional patterns.

```
```python
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

**Suggestion:**

```
```python
def mc_sample_path(P, init=0, sample_size=1_000):
    """Simulate a Markov chain sample path."""
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)
    X = X.at[0].set(init)
    n = len(P)
    # Convert rows to cumulative distributions
    P_dist = [np.cumsum(P[i, :]) for i in range(n)]
    for t in range(sample_size - 1):
        X = X.at[t+1].set(qe.random.draw(P_dist[X[t]]))
    return X
```

**Explanation:** Replaces in-place assignments with `.at[].set()` functional update pattern, which is compatible with both NumPy and JAX arrays.

### 32. qe-jax-004 — Use functional update patterns
**Location:** Line 120 / Section "Rolling our own"
**Description:** In-place array assignment `X[t+1] = qe.random.draw(...)` inside loop violates functional update pattern.

```
```python
    for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])
    return X
```

**Suggestion:**

```
```python
    for t in range(sample_size - 1):
        X = X.at[t+1].set(qe.random.draw(P_dist[X[t]]))
    return X
```

**Explanation:** Uses `.at[t+1].set()` instead of direct index assignment to follow JAX functional patterns.

### 33. qe-jax-004 — Use functional update patterns
**Location:** Line 584 / Section "Exercise mc-jax-ex1 solution"
**Description:** In-place array assignment `states[0] = 0` should use functional update pattern.

```
```python
# Simulate
np.random.seed(42)
key = jr.PRNGKey(42)
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states[0] = 0

P_np = np.array(P_growth)
```

**Suggestion:**

```
```python
# Simulate
np.random.seed(42)
key = jr.PRNGKey(42)
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states = states.at[0].set(0)

P_np = np.array(P_growth)
```

**Explanation:** Replaces direct index assignment with `.at[0].set(0)` to follow functional update pattern.

### 34. qe-jax-004 — Use functional update patterns
**Location:** Line 591 / Section "Exercise mc-jax-ex1 solution"
**Description:** In-place array assignment `states[t+1] = np.random.choice(...)` in loop violates functional update pattern.

```
```python
P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states[t+1] = np.random.choice(3, p=P_np[states[t]])

for i, label in enumerate(["Recession", "Normal", "Boom"]):
```

**Suggestion:**

```
```python
P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states = states.at[t+1].set(np.random.choice(3, p=P_np[states[t]]))

for i, label in enumerate(["Recession", "Normal", "Boom"]):
```

**Explanation:** Uses `.at[t+1].set()` to update array elements functionally rather than through in-place assignment.

### 35. qe-jax-006 — Explicit PRNG key management
**Location:** Line 247-250 / Section "Computing with JAX"
**Description:** Uses NumPy's implicit random state (`np.random.seed()` and `np.random.dirichlet()`) in a JAX-focused section instead of JAX's explicit PRNG key management.

```
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

**Suggestion:**

```
```{code-cell} ipython3
import jax.numpy as jnp
import jax.random as jr
from jax import jit

# Build a larger transition matrix
n = 50
key = jr.PRNGKey(42)
P_large = jr.dirichlet(key, jnp.ones(n), shape=(n,))
```

**Explanation:** Replaces `np.random.seed()` and `np.random.dirichlet()` with JAX's explicit PRNG key management using `jr.PRNGKey()` and `jr.dirichlet()`, which is the correct approach in JAX code.

### 36. qe-jax-006 — Explicit PRNG key management
**Location:** Line 534-546 / Section "Exercise mc-jax-ex1 solution"
**Description:** Uses NumPy's implicit random state (`np.random.seed()` and `np.random.choice()`) for simulation even though a JAX PRNG key is created but unused. This should use JAX random functions consistently.

```
```{code-cell} ipython3
# Simulate
np.random.seed(42)
key = jr.PRNGKey(42)
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states[0] = 0

P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states[t+1] = np.random.choice(3, p=P_np[states[t]])
```

**Suggestion:**

```
```{code-cell} ipython3
# Simulate
key = jr.PRNGKey(42)
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states[0] = 0

P_np = np.array(P_growth)
for t in range(n_periods - 1):
    key, subkey = jr.split(key)
    states[t+1] = jr.choice(subkey, 3, p=P_np[states[t]])
```

**Explanation:** Removes `np.random.seed()` and replaces `np.random.choice()` with JAX's `jr.choice()` using proper key splitting. This demonstrates correct explicit PRNG key management in JAX.

### 37. qe-fig-001 — Do not set figure size unless necessary
**Location:** Line 232 / Section "Convergence to stationarity"
**Description:** The code explicitly sets `figsize=(10, 6)` when creating a matplotlib figure without any comment or justification explaining why this specific size is necessary.

```
```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

psi = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("iteration")
ax.set_ylabel("probability")
```

**Suggestion:**

```
```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

psi = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots()
ax.set_xlabel("iteration")
ax.set_ylabel("probability")
```

**Explanation:** Removing the explicit `figsize` parameter allows the figure to use the default size configured in `_config.yml` for the QuantEcon lecture series, ensuring consistency across all lectures unless there is a specific, documented reason to deviate.

### 38. qe-fig-002 — Prefer code-generated figures
**Location:** Line 282-287 / Section "Convergence to stationarity"
**Description:** A static PNG image file is referenced via a `figure` directive immediately after code that generates and displays the same convergence plot using `plt.show()`. This violates the preference for code-generated figures over static image files.

```
```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: fig-convergence-plot

Convergence to Hamilton's stationary distribution
```

**Explanation:** The static figure directive should be removed entirely because the code cell immediately preceding it (lines 242-277) already generates and displays the convergence plot using matplotlib. The `plt.show()` command renders the figure inline in Jupyter notebooks, making the static image reference redundant. If a figure label is needed for cross-referencing, a `:name:` attribute can be added to the code cell's metadata instead. The caption information can be moved to markdown text following the code cell if needed.

---

## Warnings (3)

- qe-writing-007: Current text and suggested fix are identical
- ⚠️  Could not apply qe-code-003: position 1114 no longer matches current_text (likely overlapped by an earlier fix)
- qe-fig-002: Missing suggested_fix

---

## ✅ Applied Fixes (47)

> **No action required:** The following rule violations were automatically fixed in the lecture file.

### 1. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 365-366 / Section "A Larger Economic Model"
**Description:** This paragraph block contains two sentences without a blank line separating them.
**Current text:**

```
Consider a model where workers transition between income quartiles. The model uses a $4 \times 4$ transition matrix.
```

**Applied fix:**

```
Consider a model where workers transition between income quartiles.

The model uses a $4 \times 4$ transition matrix.
```

**Explanation:** The two sentences are now separated by a blank line, creating two distinct paragraph blocks.

### 2. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 254-255 / Section "Computing with JAX"
**Description:** This paragraph block contains two sentences without a blank line separating them.
**Current text:**

```
JAX provides significant performance advantages for iterative computations. Let's use it to compute stationary distributions via the Power Method.
```

**Applied fix:**

```
JAX provides significant performance advantages for iterative computations.

Let's use it to compute stationary distributions via the Power Method.
```

**Explanation:** The two sentences are now separated by a blank line, creating two distinct paragraph blocks.

### 3. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 220-221 / Section "Convergence to Stationarity"
**Description:** This paragraph block contains two sentences without a blank line separating them.
**Current text:**

```
Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition. This is a powerful result.
```

**Applied fix:**

```
Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition.

This is a powerful result.
```

**Explanation:** The two sentences are now separated by a blank line, creating two distinct paragraph blocks.

### 4. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 109-110 / Section "An Employment Model"
**Description:** This paragraph block contains two sentences without a blank line separating them.
**Current text:**

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run. These are basically the questions we want to answer in this lecture.
```

**Applied fix:**

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.

These are basically the questions we want to answer in this lecture.
```

**Explanation:** The two sentences are now separated by a blank line, creating two distinct paragraph blocks.

### 5. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 16-18 / Section "Overview"
**Description:** This paragraph block contains three sentences without blank lines separating them.
**Current text:**

```
Markov chains are one of the most useful classes of stochastic processes in economics and finance. They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state. This property is known as the **Markov property**, and it simplifies analysis considerably while still capturing rich dynamics.
```

**Applied fix:**

```
Markov chains are one of the most useful classes of stochastic processes in economics and finance.

They provide a framework for modeling systems that transition between states over time, where the future state depends only on the current state.

This property is known as the **Markov property**, and it simplifies analysis considerably while still capturing rich dynamics.
```

**Explanation:** Each sentence is now separated by a blank line, creating three distinct paragraph blocks as required by the rule.

### 6. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 389 / Section "Exercises"
**Description:** Common nouns "economic growth" are unnecessarily capitalized mid-sentence.
**Current text:**

```
Consider the following model of Economic Growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).
```

**Applied fix:**

```
Consider the following model of economic growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).
```

**Explanation:** "Economic growth" is not a proper noun and should be lowercase in narrative text.

### 7. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 365 / Section "A Larger Economic Model"
**Description:** Common noun "rate of convergence" is unnecessarily capitalized mid-sentence.
**Current text:**

```
This spectral decomposition helps us understand the Rate of Convergence to the stationary distribution.
```

**Applied fix:**

```
This spectral decomposition helps us understand the rate of convergence to the stationary distribution.
```

**Explanation:** "Rate of convergence" is a common technical term, not a proper noun.

### 8. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 136 / Section "Computing the Stationary Distribution"
**Description:** Common technical term "stationary distribution" is unnecessarily capitalized mid-sentence.
**Current text:**

```
For our employment model, we can find the Stationary Distribution analytically.
```

**Applied fix:**

```
For our employment model, we can find the stationary distribution analytically.
```

**Explanation:** "Stationary distribution" is a technical term but not a proper noun and should be lowercase in narrative text.

### 9. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 122 / Section "Theory"
**Description:** Common nouns "stochastic steady states" are unnecessarily capitalized mid-sentence.
**Current text:**

```
Hence stationary distributions represent **Stochastic Steady States**.
```

**Applied fix:**

```
Hence stationary distributions represent **stochastic steady states**.
```

**Explanation:** These are descriptive terms, not proper nouns, and should be lowercase even when emphasized.

### 10. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 59 / Section "An Employment Model"
**Description:** Common noun "transition matrix" is unnecessarily capitalized mid-sentence.
**Current text:**

```
The Transition Matrix is
```

**Applied fix:**

```
The transition matrix is
```

**Explanation:** "Transition matrix" is a common technical term, not a proper noun, and should be lowercase.

### 11. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 39 / Section "The Markov Property"
**Description:** Common nouns "current state" are unnecessarily capitalized mid-sentence.
**Current text:**

```
In other words, knowing the Current State is enough to determine probabilities for future states.
```

**Applied fix:**

```
In other words, knowing the current state is enough to determine probabilities for future states.
```

**Explanation:** "Current state" is not a proper noun and should not be capitalized mid-sentence.

### 12. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 296 / Section "A Larger Economic Model"
**Description:** Section heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
## A Larger Economic Model
```

**Applied fix:**

```
## A larger economic model
```

**Explanation:** Section headings should use sentence case, capitalizing only the first word and proper nouns.

### 13. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 268 / Section "Using jax.lax.scan for Efficient Iteration"
**Description:** Subsection heading incorrectly capitalizes "Efficient" and "Iteration" when they are not proper nouns.
**Current text:**

```
### Using jax.lax.scan for Efficient Iteration
```

**Applied fix:**

```
### Using jax.lax.scan for efficient iteration
```

**Explanation:** Subsection headings should use sentence case, capitalizing only the first word and proper nouns (jax.lax.scan is a technical term/function name and remains as-is).

### 14. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 227 / Section "The Power Method"
**Description:** Subsection heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
### The Power Method
```

**Applied fix:**

```
### The power method
```

**Explanation:** Subsection headings should use sentence case, capitalizing only the first word and proper nouns.

### 15. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 167 / Section "Convergence to Stationarity"
**Description:** Subsection heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
### Convergence to Stationarity
```

**Applied fix:**

```
### Convergence to stationarity
```

**Explanation:** Subsection headings should use sentence case, capitalizing only the first word and proper nouns.

### 16. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 150 / Section "Computing the Stationary Distribution"
**Description:** Subsection heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
### Computing the Stationary Distribution
```

**Applied fix:**

```
### Computing the stationary distribution
```

**Explanation:** Subsection headings should use sentence case, capitalizing only the first word and proper nouns.

### 17. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 126 / Section "Stationary Distributions"
**Description:** Section heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
## Stationary Distributions
```

**Applied fix:**

```
## Stationary distributions
```

**Explanation:** Section headings should use sentence case, capitalizing only the first word and proper nouns.

### 18. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 84 / Section "Simulating Markov Chains"
**Description:** Section heading incorrectly capitalizes "Chains" when only "Markov" (proper noun) should be capitalized beyond the first word.
**Current text:**

```
## Simulating Markov Chains
```

**Applied fix:**

```
## Simulating Markov chains
```

**Explanation:** Only the first word and proper nouns (Markov) should be capitalized in section headings.

### 19. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 58 / Section "An Employment Model"
**Description:** Section heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
## An Employment Model
```

**Applied fix:**

```
## An employment model
```

**Explanation:** Section headings should use sentence case, capitalizing only the first word and proper nouns.

### 20. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 42 / Section "The Markov Property"
**Description:** Subsection heading incorrectly capitalizes "Property" when only "Markov" (proper noun) should be capitalized beyond the first word.
**Current text:**

```
### The Markov Property
```

**Applied fix:**

```
### The Markov property
```

**Explanation:** Only the first word and proper nouns (Markov) should be capitalized in subsection headings.

### 21. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 32 / Section "Stochastic Matrices"
**Description:** Subsection heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
### Stochastic Matrices
```

**Applied fix:**

```
### Stochastic matrices
```

**Explanation:** Subsection headings should use sentence case, capitalizing only the first word and proper nouns.

### 22. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 30 / Section "Definitions and Setup"
**Description:** Section heading capitalizes all words instead of only the first word and proper nouns.
**Current text:**

```
## Definitions and Setup
```

**Applied fix:**

```
## Definitions and setup
```

**Explanation:** Section headings should use sentence case, capitalizing only the first word and proper nouns.

### 23. qe-writing-005 — Use bold for definitions, italic for emphasis
**Location:** Line 341 / Section "A larger economic model"
**Description:** The label "Note:" uses bold formatting for emphasis/attention rather than for a definition. According to the rule, emphasis should use italic, while bold is reserved for definitions.
**Current text:**

```
**Note:** The stationary distribution gives us the long-run fraction of the population in each income quartile.
```

**Applied fix:**

```
*Note:* The stationary distribution gives us the long-run fraction of the population in each income quartile.
```

**Explanation:** This changes the formatting from bold to italic, which is the appropriate format for emphasis according to the style rule. The label is drawing attention to important information rather than defining a term.

### 24. qe-writing-005 — Use bold for definitions, italic for emphasis
**Location:** Line 339 / Section "A larger economic model"
**Description:** The label "Important:" uses bold formatting for emphasis/attention rather than for a definition. According to the rule, emphasis should use italic, while bold is reserved for definitions.
**Current text:**

```
**Important:** Make sure you understand the connection between the transition matrix and the economic model before proceeding.
```

**Applied fix:**

```
*Important:* Make sure you understand the connection between the transition matrix and the economic model before proceeding.
```

**Explanation:** This changes the formatting from bold to italic, which is the appropriate format for emphasis according to the style rule. The label is drawing attention to the paragraph rather than defining a term.

### 25. qe-math-001 — Prefer UTF-8 unicode for simple parameter mentions, be consistent
**Location:** Line 72 / Section "An employment model"
**Description:** The parameters α and β are mentioned in narrative text using inline math delimiters (`$\alpha$` and `$\beta$`), but this sentence contains no mathematical expressions. According to the rule, simple parameter mentions in narrative text should use UTF-8 unicode characters (α, β) to improve readability and reduce visual clutter.
**Current text:**

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.
```

**Applied fix:**

```
Once we have the values of α and β, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.
```

**Explanation:** This sentence is narrative text that simply mentions the parameter names without any mathematical expressions or formulas. Using UTF-8 unicode characters (α, β) instead of inline math (`$\alpha$`, `$\beta$`) improves readability and follows the preferred style for simple parameter mentions.

### 26. qe-math-002 — Use \top for transpose notation
**Location:** Line 373 / Section "A larger economic model"
**Description:** The matrix transpose uses `^T` notation instead of `^\top` in inline math discussing eigenvalues.
**Current text:**

```
The eigenvalues of $P^T$ determine how quickly the distribution converges.
```

**Applied fix:**

```
The eigenvalues of $P^\top$ determine how quickly the distribution converges.
```

**Explanation:** Replacing `P^T` with `P^\top` follows the required transpose notation standard using `\top`.

### 27. qe-math-002 — Use \top for transpose notation
**Location:** Line 367 / Section "A larger economic model"
**Description:** The matrix transpose uses `^T` notation instead of `^\top` in the spectral decomposition equation.
**Current text:**

```
$$
\mathbf{P} = \mathbf{L} \mathbf{D} \mathbf{L}^T
$$
```

**Applied fix:**

```
$$
\mathbf{P} = \mathbf{L} \mathbf{D} \mathbf{L}^\top
$$
```

**Explanation:** Replacing `\mathbf{L}^T` with `\mathbf{L}^\top` follows the required transpose notation standard using `\top`.

### 28. qe-math-003 — Use square brackets for matrix notation
**Location:** Line 70 / Section "An employment model"
**Description:** The transition matrix P uses `\begin{pmatrix}` (parentheses) instead of the required `\begin{bmatrix}` (square brackets) for matrix notation.
**Current text:**

```
$$
P = \begin{pmatrix}
1 - \alpha & \alpha \\
\beta & 1 - \beta
\end{pmatrix}
$$
```

**Applied fix:**

```
$$
P = \begin{bmatrix}
1 - \alpha & \alpha \\
\beta & 1 - \beta
\end{bmatrix}
$$
```

**Explanation:** Replacing `\begin{pmatrix}...\end{pmatrix}` with `\begin{bmatrix}...\end{bmatrix}` changes the matrix delimiters from parentheses to square brackets, which is the required notation per the style rule qe-math-003.

### 29. qe-math-004 — Do not use bold face for matrices or vectors
**Location:** Line 393 / Section "A larger economic model"
**Description:** Uses `\mathbf{L}` and `\mathbf{D}` for matrices in inline math, which violates the rule against bold face formatting.
**Current text:**

```
where $\mathbf{L}$ is a lower triangular matrix and $\mathbf{D}$ is diagonal.
```

**Applied fix:**

```
where $L$ is a lower triangular matrix and $D$ is diagonal.
```

**Explanation:** Removes `\mathbf{}` and uses plain letters L and D for the matrices, as required by the style rule.

### 30. qe-math-004 — Do not use bold face for matrices or vectors
**Location:** Line 389-391 / Section "A larger economic model"
**Description:** Uses `\mathbf{}` for multiple matrices (P, L, D) in display math equation, which violates the rule against bold face formatting.
**Current text:**

```
$$
\mathbf{P} = \mathbf{L} \mathbf{D} \mathbf{L}^\top
$$
```

**Applied fix:**

```
$$
P = L D L^\top
$$
```

**Explanation:** Removes all `\mathbf{}` commands and uses plain letters for matrices P, L, and D, as required by the style rule.

### 31. qe-math-004 — Do not use bold face for matrices or vectors
**Location:** Line 387 / Section "A larger economic model"
**Description:** Uses `\mathbf{P}` for a matrix in inline math, which violates the rule against bold face formatting for matrices.
**Current text:**

```
The income transition matrix $\mathbf{P}$ can be decomposed as
```

**Applied fix:**

```
The income transition matrix $P$ can be decomposed as
```

**Explanation:** Removes `\mathbf{}` and uses plain letter $P$ for the matrix, as required by the style rule.

### 32. qe-math-007 — Use automatic equation numbering, not manual tags
**Location:** Line 165 / Section "Stationary distributions"
**Description:** The equation defining stationary distribution uses manual numbering with `\tag{1}` instead of automatic equation numbering with a label.
**Current text:**

```
$$
\psi^* = \psi^* P \tag{1}
$$
```

**Applied fix:**

```
$$
\psi^* = \psi^* P
$$ (stationary-dist)
```

**Explanation:** This removes the manual `\tag{1}` and replaces it with MyST's automatic equation numbering syntax using a label `(stationary-dist)` after the closing `$$`. This equation can then be referenced elsewhere using `{eq}`stationary-dist`` if needed, and the numbering will be handled automatically by the documentation system.

### 33. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 148-151 / Section "Computing the stationary distribution"
**Description:** Print statement uses spelled-out `alpha` and `beta` variable names instead of Unicode symbols α and β.
**Current text:**

```
```{code-cell} ipython3
mc = qe.MarkovChain(P)
psi_star = mc.stationary_distributions[0]
print(f"Stationary distribution: {psi_star}")
print(f"Theoretical: [{beta/(alpha+beta):.4f}, {alpha/(alpha+beta):.4f}]")
```

**Applied fix:**

```
```{code-cell} ipython3
mc = qe.MarkovChain(P)
psi_star = mc.stationary_distributions[0]
print(f"Stationary distribution: {psi_star}")
print(f"Theoretical: [{β/(α+β):.4f}, {α/(α+β):.4f}]")
```

**Explanation:** Updating variable references to use α and β maintains consistency with the corrected variable definitions and improves code readability by matching mathematical notation.

### 34. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 87-93 / Section "An employment model"
**Description:** Variable names use spelled-out `alpha` and `beta` instead of Unicode symbols α and β, which are standard in economic notation and improve code readability.
**Current text:**

```
```{code-cell} ipython3
alpha = 0.3    # probability of finding a job
beta = 0.2     # probability of losing a job

P = np.array([[1 - alpha, alpha],
              [beta, 1 - beta]])

print(P)
```

**Applied fix:**

```
```{code-cell} ipython3
α = 0.3    # probability of finding a job
β = 0.2     # probability of losing a job

P = np.array([[1 - α, α],
              [β, 1 - β]])

print(P)
```

**Explanation:** Replacing `alpha` with `α` and `beta` with `β` makes the code notation consistent with the mathematical formulas in the lecture and follows QuantEcon style guidelines for Greek letters.

### 35. qe-code-003 — Package installation at lecture top
**Location:** Line 42-44 / Section "Overview"
**Description:** The installation cell for `quantecon` is missing the required `tags: [hide-output]` metadata to suppress verbose installation output.
**Current text:**

```
```{code-cell} ipython3
!pip install quantecon
```

**Applied fix:**

```
```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon
```

**Explanation:** Adding the `tags: [hide-output]` metadata suppresses the verbose pip installation output, making the rendered lecture cleaner and more readable.

### 36. qe-jax-002 — Use NamedTuple for model parameters
**Location:** Line 138-154 / Section "The power method"
**Description:** The `StationarySolver` class uses a mutable class structure primarily for storing configuration parameters (P, tol, max_iter) with a single computational method. This violates the rule requiring NamedTuple for parameter storage with factory functions for validation.
**Current text:**

```
```{code-cell} ipython3
class StationarySolver:
    """Solver for stationary distributions using the power method."""
    
    def __init__(self, P, tol=1e-10, max_iter=10_000):
        self.P = P
        self.tol = tol
        self.max_iter = max_iter
        self.n = P.shape[0]
    
    def solve(self):
        psi = jnp.ones(self.n) / self.n
        for i in range(self.max_iter):
            psi_new = psi @ self.P
            if jnp.max(jnp.abs(psi_new - psi)) < self.tol:
                return psi_new, i + 1
            psi = psi_new
        return psi, self.max_iter
```

**Applied fix:**

```
```{code-cell} ipython3
from typing import NamedTuple

class StationaryParams(NamedTuple):
    """Parameters for stationary distribution solver."""
    P: jnp.ndarray
    tol: float = 1e-10
    max_iter: int = 10_000

def create_stationary_solver(P, tol=1e-10, max_iter=10_000):
    """Factory function to create solver parameters with validation."""
    if tol <= 0:
        raise ValueError("Tolerance must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    return StationaryParams(P=P, tol=tol, max_iter=max_iter)

def solve_stationary(params: StationaryParams):
    """Solve for stationary distribution using the power method."""
    n = params.P.shape[0]
    psi = jnp.ones(n) / n
    for i in range(params.max_iter):
        psi_new = psi @ params.P
        if jnp.max(jnp.abs(psi_new - psi)) < params.tol:
            return psi_new, i + 1
        psi = psi_new
    return psi, params.max_iter
```

**Explanation:** This fix separates immutable parameter storage (using NamedTuple) from the computational logic. The factory function `create_stationary_solver` provides parameter validation, and `solve_stationary` is a pure function that takes parameters and returns results. This follows the preferred pattern of using NamedTuple for configuration storage with factory functions.

### 37. qe-fig-003 — No matplotlib embedded titles
**Location:** Line 229 / Section "Convergence to stationarity"
**Description:** The code uses `ax.set_title()` to embed a title directly in the matplotlib figure. This violates the rule requiring titles to be added via MyST metadata or figure directives instead. This code is not within an exercise or solution context, so the exception does not apply.
**Current text:**

```
```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

psi = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Convergence to Stationary Distribution")
ax.set_xlabel("Iteration")
ax.set_ylabel("Probability")

psi_history = [psi.copy()]
for t in range(30):
    psi = psi @ P_hamilton
    psi_history.append(psi.copy())

psi_history = np.array(psi_history)

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

**Applied fix:**

```
```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

psi = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Iteration")
ax.set_ylabel("Probability")

psi_history = [psi.copy()]
for t in range(30):
    psi = psi @ P_hamilton
    psi_history.append(psi.copy())

psi_history = np.array(psi_history)

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

**Explanation:** The fix removes the `ax.set_title("Convergence to Stationary Distribution")` line. The title should instead be added using MyST figure directive metadata (as shown in the subsequent `{figure}` directive that already exists) or through cell metadata if using `mystnb`.

### 38. qe-fig-004 — Caption formatting conventions
**Location:** Line 329 / Section "Convergence to stationarity"
**Description:** The figure caption uses Title Case capitalization and is excessively long (12 words vs. maximum 5-6 words). It should use sentence case (lowercase except first letter and proper nouns) and be more concise.
**Current text:**

```
Convergence of Marginal Distributions to the Stationary Distribution for Hamilton's Recession Model
```

**Applied fix:**

```
Convergence to Hamilton's stationary distribution
```

**Explanation:** This fix resolves both violations by (1) converting to sentence case while preserving "Hamilton's" as a proper noun, and (2) reducing the caption from 12 words to 5 words, meeting the conciseness requirement.

### 39. qe-fig-005 — Descriptive figure names for cross-referencing
**Location:** Line 289 / Section "Convergence to stationarity"
**Description:** The figure name `convergence-plot` does not follow the required `fig-` prefix convention. According to the rule, all figure names should follow the pattern `fig-description` using lowercase with hyphens.
**Current text:**

```
```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: convergence-plot

Convergence to Hamilton's stationary distribution
```

**Applied fix:**

```
```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: fig-convergence-plot

Convergence to Hamilton's stationary distribution
```

**Explanation:** Changing the name from `convergence-plot` to `fig-convergence-plot` adds the required `fig-` prefix while maintaining the descriptive nature and hyphen-separated lowercase format. Note: The corresponding `numref` reference on line 295 (`{numref}\`convergence-plot\``) should also be updated to `{numref}\`fig-convergence-plot\`` to maintain consistency, though cross-reference updates may be outside the scope of this specific rule check.

### 40. qe-fig-006 — Lowercase axis labels
**Location:** Line 249 / Section "Convergence to stationarity"
**Description:** The y-axis label uses title case "Probability" instead of lowercase as required by the style guide.
**Current text:**

```
ax.set_ylabel("Probability")
```

**Applied fix:**

```
ax.set_ylabel("probability")
```

**Explanation:** Axis labels should be lowercase unless they are proper nouns. "Probability" is not a proper noun and should be "probability".

### 41. qe-fig-006 — Lowercase axis labels
**Location:** Line 248 / Section "Convergence to stationarity"
**Description:** The x-axis label uses title case "Iteration" instead of lowercase as required by the style guide.
**Current text:**

```
ax.set_xlabel("Iteration")
```

**Applied fix:**

```
ax.set_xlabel("iteration")
```

**Explanation:** Axis labels should be lowercase unless they are proper nouns. "Iteration" is not a proper noun and should be "iteration".

### 42. qe-ref-001 — Use correct citation style
**Location:** Line 448 / Section "Exercises"
**Description:** Manual citation formatting where the author name "Hamilton" is explicitly written in the text and then followed by a `{cite}` reference. This creates redundancy and violates the citation style rule against manual formatting.
**Current text:**

```
The transition matrix is given by Hamilton {cite}`Hamilton2005`.
```

**Applied fix:**

```
The transition matrix is given by {cite:t}`Hamilton2005`.
```

**Explanation:** Using `{cite:t}` allows the citation itself to provide the author name as part of the sentence flow, eliminating the manual "Hamilton" mention. This follows the correct in-text citation pattern where the author name is part of the sentence structure ("given by [Author]"). Alternatively, the sentence could be restructured to "The transition matrix is from {cite}`Hamilton2005`" for a parenthetical style, but the in-text approach better preserves the original meaning.

### 43. qe-link-002 — Use doc links for cross-series references
**Location:** Line 441 / Section "A larger economic model"
**Description:** Markdown link with direct URL to intro series instead of using {doc} link with intersphinx prefix
**Current text:**

```
Also check [this lecture](https://intro.quantecon.org/markov_chains_I.html) for introductory material.
```

**Applied fix:**

```
Also check {doc}`this lecture<intro:markov_chains_I>` for introductory material.
```

**Explanation:** The URL https://intro.quantecon.org/ corresponds to the intro series and should use the `{doc}` syntax with the `intro:` prefix and custom title format for proper intersphinx cross-referencing.

### 44. qe-link-002 — Use doc links for cross-series references
**Location:** Line 439 / Section "A larger economic model"
**Description:** Direct URL to intermediate series lecture instead of using {doc} link with intersphinx prefix
**Current text:**

```
For more details, see https://python.quantecon.org/wealth_dynamics.html on wealth distribution.
```

**Applied fix:**

```
For more details, see {doc}`intermediate:wealth_dynamics` on wealth distribution.
```

**Explanation:** The URL https://python.quantecon.org/ corresponds to the intermediate series and should use the `{doc}` syntax with the `intermediate:` prefix for proper intersphinx cross-referencing.

### 45. qe-admon-001 — Use gated syntax for executable code in exercises
**Location:** Section "Exercises" (Exercise mc-jax-ex2)
**Description:** This exercise contains nested directives (`solution-start`/`solution-end`) with executable code cells, but uses standard `{exercise}` syntax instead of gated syntax.
**Current text:**

```
```{exercise}
:label: mc-jax-ex2

Write a function using `jax.lax.fori_loop` that computes $\psi P^t$ for a given initial distribution $\psi$ and transition matrix $P$, returning the distribution at each time step. Compare the performance with a pure Python loop.
```

**Applied fix:**

```
```{exercise-start}
:label: mc-jax-ex2
```

**Explanation:** Converting to gated syntax (`exercise-start`/`exercise-end`) ensures proper handling of nested solution directives containing executable code cells.

### 46. qe-admon-001 — Use gated syntax for executable code in exercises
**Location:** Section "Exercises" (Exercise mc-jax-ex1)
**Description:** This exercise contains nested directives (`solution-start`/`solution-end`) with executable code cells, but uses standard `{exercise}` syntax instead of gated syntax.
**Current text:**

```
````{exercise}
:label: mc-jax-ex1

Consider the following model of economic growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).

The transition matrix is given by {cite:t}`Hamilton2005`.

Compute the stationary distribution and simulate 10,000 periods to verify that the time averages match the stationary probabilities.

Use JAX for the computation.
```

**Applied fix:**

```
```{exercise-start}
:label: mc-jax-ex1
```

**Explanation:** Converting to gated syntax (`exercise-start`/`exercise-end`) allows proper parsing of nested directives and executable code cells within the exercise block.

### 47. qe-admon-003 — Use tick count management for nested directives
**Location:** Line 502 / Section "Exercises"
**Description:** The `exercise-start` directive for `mc-jax-ex2` uses 3 backticks but contains nested `solution-start` and `code-cell` directives that also use 3 backticks. The outer directive should use more ticks than nested directives. Since this exercise contains a solution block (which uses 3 ticks) containing a code-cell block (which uses 3 ticks), the exercise-start should use at least 4 ticks to properly contain the nested structure.
**Current text:**

```
```{exercise-start}
:label: mc-jax-ex2
```

**Applied fix:**

```
````{exercise-start}
:label: mc-jax-ex2
```

**Explanation:** Changing the exercise-start directive to use 4 backticks ensures proper nesting hierarchy: the outer exercise block (4 ticks) properly contains the solution-start block (3 ticks), which in turn contains the code-cell block (3 ticks). This follows the standard pattern where outer directives use more ticks than nested directives to avoid parsing ambiguity.
