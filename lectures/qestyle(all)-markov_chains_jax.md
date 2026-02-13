# Style Guide Report: markov_chains_jax.md

- **Date:** 2026-02-13 16:07
- **Version:** qestyle v0.7.0
- **Issues found:** 94
- **Mode:** fix (rule violations applied to file)

## 📝 Style Suggestions (35)

> **Action required:** These suggestions require human review and judgment.

### 1. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 48 / Section "Definitions and setup"
**Description:** The phrase "It is not difficult to show that" is unnecessarily verbose and adds no value. The statement can be made directly.

```
It is not difficult to show that if $P$ is a stochastic matrix, then so is $P^k$ for all $k \in \mathbb{N}$.
```

**Suggestion:**

```
If $P$ is a stochastic matrix, then so is $P^k$ for all $k \in \mathbb{N}$.
```

**Explanation:** Removing the hedging phrase makes the statement direct and clearer without losing any meaning.

### 2. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 82 / Section "An employment model"
**Description:** The word "basically" is filler that adds no value to the sentence.

```
These are basically the questions we want to answer in this lecture.
```

**Suggestion:**

```
These are the questions we want to answer in this lecture.
```

**Explanation:** Removing "basically" makes the statement more direct without changing its meaning.

### 3. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 79 / Section "An employment model"
**Description:** This sentence is overly long (36 words) and has a complex structure that reduces clarity. It lists multiple questions in an unwieldy way.

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.
```

**Suggestion:**

```
Once we have $\alpha$ and $\beta$, we can address questions like: What is the average duration of unemployment? What fraction of time does a worker spend unemployed in the long run?
```

**Explanation:** Simplifying the opening phrase and breaking the questions into clearer segments improves readability.

### 4. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 206 / Section "Convergence to stationarity"
**Description:** This sentence is redundant. The caption reference immediately before it already indicates what the figure shows.

```
The figure above shows how the distribution converges over time.
```

**Suggestion:**

```
[Remove this sentence entirely]
```

**Explanation:** The sentence adds no new information beyond what the figure caption and the surrounding context already convey.

### 5. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 292 / Section "A larger economic model"
**Description:** This link goes to the main QuantEcon page rather than specific content about income dynamics models, providing no actionable value to readers.

```
Click [here](https://quantecon.org) to learn more about income dynamics models.
```

**Suggestion:**

```
[Remove this sentence entirely]
```

**Explanation:** The link doesn't direct readers to relevant content about income dynamics models, making it uninformative.

### 6. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 318 / Section "A larger economic model"
**Description:** This admonition is vague and doesn't provide specific, actionable guidance. It doesn't add value beyond a generic reminder.

```
**Important:** Make sure you understand the connection between the transition matrix and the economic model before proceeding.
```

**Suggestion:**

```
[Remove this admonition entirely]
```

**Explanation:** The statement is too vague to be helpful. If understanding this connection is important, the lecture should explain it explicitly rather than issuing a generic warning.

### 7. qe-writing-002 — Keep writing clear, concise, and valuable
**Location:** Line 338 / Section "A larger economic model"
**Description:** This sentence is redundant with the immediately preceding sentence, which already states that eigenvalues determine convergence speed.

```
The largest eigenvalue is always 1 (corresponding to the stationary distribution), and the second-largest eigenvalue determines the rate of convergence.
```

**Suggestion:**

```
The largest eigenvalue is always 1 (corresponding to the stationary distribution).
```

**Explanation:** The information about the second-largest eigenvalue determining convergence rate repeats what the previous sentence already conveyed. Keeping only the new information about the largest eigenvalue adds value without redundancy.

### 8. qe-writing-003 — Maintain logical flow
**Location:** Line 308-314 / Section "A larger economic model"
**Description:** Three consecutive external reference links appear without integration into the narrative flow. These tangential references distract from the main content about the income dynamics model and break the logical progression.

```
Click [here](https://quantecon.org) to learn more about income dynamics models.

For more details, see https://python.quantecon.org/wealth_dynamics.html on wealth distribution.

Also check [this lecture](https://intro.quantecon.org/markov_chains_I.html) for introductory material.
```

**Suggestion:**

```
This model is an example of income dynamics modeling, which we will explore in more detail in later lectures.
```

**Explanation:** The fix removes the distracting tangential links and replaces them with a single forward-looking statement that maintains narrative focus while acknowledging related content exists elsewhere.

### 9. qe-writing-003 — Maintain logical flow
**Location:** Line 338-352 / Section "A larger economic model"
**Description:** The spectral decomposition discussion is introduced abruptly without transition or motivation. The text jumps from computing stationary distributions to advanced matrix decomposition theory without explaining why this is relevant or how it connects to the previous content.

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
To understand how quickly the distribution converges to steady state, we can examine the eigenvalues of $P^T$. The largest eigenvalue is always 1 (corresponding to the stationary distribution), and the second-largest eigenvalue determines the convergence rate.
```

**Explanation:** This fix removes the unmotivated and technically questionable matrix decomposition, providing instead a direct and properly motivated transition to eigenvalue analysis that connects to the previously discussed concept of convergence.

### 10. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 79-95 / Section "An employment model"
**Description:** This section describes a two-state Markov chain with transitions between unemployment and employment states, but presents only the mathematical transition matrix without any visual diagram. A state transition diagram would make the dynamics immediately clear and is a standard visualization for Markov chains.

```
## An employment model

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
## An employment model

Consider a worker who at any given time $t$ is either unemployed (state 0) or employed (state 1).

Suppose that, over a one month period,

1. An unemployed worker finds a job with probability $\alpha \in (0, 1)$.
1. An employed worker loses her job and becomes unemployed with probability $\beta \in (0, 1)$.
```

**Explanation:** Adding a state transition diagram would visualize the two states (unemployed/employed) and the transition probabilities between them, making the model structure immediately clear. This is a standard practice in presenting Markov chain models, as shown in the reference example.

---

### 11. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 186-192 / Section "Stationary distributions"
**Description:** An important theoretical result (theorem) is presented with only bold formatting rather than using a proper admonition block. Theorems are key mathematical results that should be visually distinguished to emphasize their importance.

```
**Theorem.** If $P$ is both aperiodic and irreducible, then

1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.

For a proof, see, for example, theorem 5.2 of {cite}`haggstrom2002finite`.
```

**Suggestion:**

```
```{prf:theorem} Convergence to Stationary Distribution
:label: stationary-convergence

If $P$ is both aperiodic and irreducible, then

1. $P$ has exactly one stationary distribution $\psi^*$.
1. For any initial distribution $\psi_0$, we have $\| \psi_0 P^t - \psi^* \| \to 0$ as $t \to \infty$.

For a proof, see, for example, theorem 5.2 of {cite}`haggstrom2002finite`.
```

**Explanation:** Using the `{prf:theorem}` directive provides proper visual formatting with a colored box and label, making this critical theoretical result stand out and easier to reference. This follows MyST best practices for mathematical content.

---

### 12. qe-writing-007 — Use visual elements to enhance understanding
**Location:** Line 400-408 / Section "A larger economic model"
**Description:** Important notes and warnings are formatted as bold inline text rather than using proper admonition blocks. This reduces their visual prominence and makes them easy to overlook.

```
**Important:** Make sure you understand the connection between the transition matrix and the economic model before proceeding.

**Note:** The stationary distribution gives us the long-run fraction of the population in each income quartile.
```

**Suggestion:**

```
```{important}
Make sure you understand the connection between the transition matrix and the economic model before proceeding.
```

**Explanation:** Using proper `{important}` and `{note}` admonition blocks provides visual distinction through colored boxes and icons, making these callouts more prominent and improving the document's visual hierarchy. This is consistent with MyST Markdown best practices for emphasizing key information.

### 13. qe-math-009 — Choose simplicity in mathematical notation
**Location:** Line 43 / Section "Stochastic matrices"
**Description:** Uses blackboard bold $\mathbb{N}$ for natural numbers when simpler alternatives exist (e.g., writing out "positive integers" or using $k \geq 1$).

```
It is not difficult to show that if $P$ is a stochastic matrix, then so is $P^k$ for all $k \in \mathbb{N}$.
```

**Suggestion:**

```
It is not difficult to show that if $P$ is a stochastic matrix, then so is $P^k$ for all positive integers $k$.
```

**Explanation:** Removing the decorative blackboard bold notation $\mathbb{N}$ in favor of plain text "positive integers" simplifies the notation without losing clarity. Alternatively, $k \geq 1$ or $k = 1, 2, 3, \ldots$ would also be simpler.

### 14. qe-math-009 — Choose simplicity in mathematical notation
**Location:** Line 51 / Section "The Markov property"
**Description:** Uses blackboard bold $\mathbb{P}$ for probability when the simpler standard notation $\Pr$ would suffice.

```
$$
\mathbb{P}\{X_{t+1} = y \mid X_t\} = \mathbb{P}\{X_{t+1} = y \mid X_t, X_{t-1}, \ldots\}
$$
```

**Suggestion:**

```
$$
\Pr\{X_{t+1} = y \mid X_t\} = \Pr\{X_{t+1} = y \mid X_t, X_{t-1}, \ldots\}
$$
```

**Explanation:** The notation $\Pr$ is a standard, simpler alternative to $\mathbb{P}$ that avoids decorative blackboard bold formatting while remaining clear and conventional for probability.

### 15. qe-math-009 — Choose simplicity in mathematical notation
**Location:** Line 57 / Section "The Markov property"
**Description:** Uses blackboard bold $\mathbb{P}$ for probability when the simpler standard notation $\Pr$ would suffice.

```
$$
P(x, y) := \mathbb{P}\{X_{t+1} = y \mid X_t = x\} \qquad (x, y \in S)
$$
```

**Suggestion:**

```
$$
P(x, y) := \Pr\{X_{t+1} = y \mid X_t = x\} \qquad (x, y \in S)
$$
```

**Explanation:** The notation $\Pr$ is a standard, simpler alternative to $\mathbb{P}$ that avoids decorative blackboard bold formatting while remaining clear and conventional for probability.

### 16. qe-code-001 — Follow PEP8 unless closer to mathematical notation
**Location:** Line 200 (approximately) / Section "Computing the stationary distribution"
**Description:** Mathematical operators `/` and `+` lack proper spacing around them. PEP8 requires spaces around binary operators (`a / b`, `a + b`), and there is no mathematical notation justification for omitting them in this context.

```
print(f"Theoretical: [{beta/(alpha+beta):.4f}, {alpha/(alpha+beta):.4f}]")
```

**Suggestion:**

```
print(f"Theoretical: [{beta / (alpha + beta):.4f}, {alpha / (alpha + beta):.4f}]")
```

**Explanation:** This fix adds the required spaces around the `/` and `+` operators to comply with PEP8 conventions. The operators appear within f-string expressions performing mathematical calculations, where standard spacing rules apply. This makes the code more readable while maintaining the same functionality.

### 17. qe-code-004 — Use quantecon Timer context manager
**Location:** Line 163-168 / Section "Using QuantEcon"
**Description:** Manual timing pattern using `time.time()` start/end variables with explicit elapsed time calculation and print statement, rather than using the modern `qe.Timer()` context manager.

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
with qe.Timer():
    X = mc.simulate(ts_length=1_000_000)
```

**Explanation:** Replacing the manual timing pattern with `qe.Timer()` context manager simplifies the code, removes the need for `time` import, and uses the modern QuantEcon timing utility. The Timer will automatically print the elapsed time upon exiting the context.

### 18. qe-code-004 — Use quantecon Timer context manager
**Location:** Line 399-404 / Section "JIT-compiled version"
**Description:** Manual timing pattern using `time.time()` start/end variables with explicit elapsed time calculation and print statement, rather than using the modern `qe.Timer()` context manager.

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
with qe.Timer():
    result = power_method_jax(P_large)
```

**Explanation:** Replacing the manual timing pattern with `qe.Timer()` context manager simplifies the code, removes the need for `time` import, and uses the modern QuantEcon timing utility. The Timer will automatically print the elapsed time upon exiting the context.

### 19. qe-code-005 — Use quantecon timeit for benchmarking
**Location:** Line 169 / Section "Using QuantEcon"
**Description:** Manual timing using `time.time()` for a single run instead of using `qe.timeit()` for statistical performance analysis across multiple runs.

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
timing_result = qe.timeit(lambda: mc.simulate(ts_length=1_000_000), number=10)
print(f"QuantEcon simulation: {timing_result.average:.4f} seconds (avg over {timing_result.number} runs)")
```

**Explanation:** Using `qe.timeit()` provides statistical timing over multiple runs rather than a single measurement, which is more reliable for benchmarking. The lambda function allows passing the method call with arguments.

### 20. qe-code-005 — Use quantecon timeit for benchmarking
**Location:** Line 360 / Section "JIT-compiled version"
**Description:** Manual timing using `time.time()` instead of using `qe.timeit()` for statistical performance analysis.

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
print(f"JAX power method: {timing_result.average:.4f} seconds (avg over {timing_result.number} runs)")
```

**Explanation:** Replacing manual timing with `qe.timeit()` provides more accurate statistical performance measurements across multiple runs, which is especially important for JIT-compiled functions where the first run may include compilation overhead.

### 21. qe-code-005 — Use quantecon timeit for benchmarking
**Location:** Line 569 / Section "Exercise mc-jax-ex2 solution"
**Description:** Using Jupyter magic command `%timeit` instead of `qe.timeit()` for benchmarking.

```
```{code-cell} ipython3
%timeit iterate_distribution(P_test, psi_0, 1000).block_until_ready()
```

**Suggestion:**

```
```{code-cell} ipython3
timing_result = qe.timeit(lambda: iterate_distribution(P_test, psi_0, 1000).block_until_ready(), number=100)
print(f"Timing: {timing_result.average:.6f} seconds (avg over {timing_result.number} runs)")
```

**Explanation:** Using `qe.timeit()` instead of the `%timeit` magic command ensures consistent benchmarking methodology throughout QuantEcon lectures and provides programmatic access to timing results.

### 22. qe-jax-001 — Use functional programming patterns
**Location:** Line 323 / Section "Computing with JAX"
**Description:** The `update_distribution` function uses a mutable Python list with `.append()` operations when working with JAX arrays, violating the functional programming pattern that JAX encourages.

```
```{code-cell} ipython3
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

**Suggestion:**

```
```{code-cell} ipython3
def update_distribution(psi, P):
    """Update the distribution by one step."""
    return psi @ P
```

**Explanation:** This fix eliminates the mutable list operations and uses pure JAX matrix multiplication, which is both more functional and more efficient.

### 23. qe-jax-001 — Use functional programming patterns
**Location:** Line 477 / Section "Exercises"
**Description:** The exercise solution modifies the `states` array in place using index assignment (`states[0] = 0` and `states[t+1] = ...`), which violates functional programming patterns.

```
```{code-cell} ipython3
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
key = jr.PRNGKey(42)
n_periods = 10_000

# Use functional approach with list comprehension
P_np = np.array(P_growth)
states = [0]  # initial state
np.random.seed(42)
for t in range(n_periods - 1):
    states.append(np.random.choice(3, p=P_np[states[-1]]))
states = np.array(states)
```

**Explanation:** While still using NumPy for this simulation, this approach builds the array functionally by collecting results rather than mutating an existing array in place.

### 24. qe-jax-001 — Use functional programming patterns
**Location:** Line 477 / Section "Exercises"
**Description:** The code uses `np.random.seed(42)` which modifies global state, creating a side effect that violates functional programming principles.

```
```{code-cell} ipython3
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
key = jr.PRNGKey(42)
n_periods = 10_000

# Use JAX's functional random number generation
def simulate_chain(key, P, n_periods, init_state=0):
    states = [init_state]
    for t in range(n_periods - 1):
        key, subkey = jr.split(key)
        states.append(int(jr.choice(subkey, 3, p=P[states[-1]])))
    return jnp.array(states)

states = simulate_chain(key, P_growth, n_periods, 0)
```

**Explanation:** This fix uses JAX's functional random number generation with explicit key splitting, eliminating the global state modification and embracing functional patterns throughout.

### 25. qe-jax-003 — Use generate_path for sequence generation
**Location:** Line 271-281 / Section "Computing with JAX"
**Description:** The code implements a custom `jax.lax.scan` pattern to generate a sequence of distribution states over time. This duplicates the functionality that the standardized `generate_path` function is designed to provide, reducing code consistency and reusability.

```
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

**Suggestion:**

```
```{code-cell} ipython3
from functools import partial

@partial(jax.jit, static_argnames=['f', 'num_steps'])
def generate_path(f, initial_state, num_steps, **kwargs):
    def update_wrapper(state, t):
        next_state = f(state, t, **kwargs)
        return next_state, state
    _, path = jax.lax.scan(update_wrapper, initial_state, jnp.arange(num_steps))
    return path.T

def update_distribution(ψ, t, P):
    """Update distribution by one step."""
    return ψ @ P

@jit
def power_method_scan(P, num_steps=1000):
    n = P.shape[0]
    ψ_0 = jnp.ones(n) / n
    ψ_history = generate_path(update_distribution, ψ_0, num_steps, P=P)
    return ψ_history[-1], ψ_history
```

**Explanation:** This fix replaces the custom scan implementation with the standardized `generate_path` function pattern. The distribution update logic is extracted into a separate `update_distribution` function that follows the signature expected by `generate_path` (state, t, **kwargs). This improves code consistency, makes the pattern reusable across the codebase, and follows JAX best practices for sequence generation.

### 26. qe-jax-005 — Use jax.lax for control flow
**Location:** Line 264 / Section "Computing with JAX"
**Description:** The `power_method_jax` function is decorated with `@jit` but uses a Python `for` loop for fixed iterations. This should be replaced with `jax.lax.fori_loop` for proper JAX compilation and performance.

```
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

**Suggestion:**

```
```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax import jit

@jit
def power_method_jax(P, tol=1e-10, max_iter=10_000):
    n = P.shape[0]
    ψ = jnp.ones(n) / n
    
    def body_fn(i, ψ):
        ψ_new = ψ @ P
        return ψ_new
    
    # Use jax.lax.fori_loop for fixed iterations
    ψ = jax.lax.fori_loop(0, 1000, body_fn, ψ)
    
    return ψ
```

**Explanation:** This fix replaces the Python `for` loop with `jax.lax.fori_loop`, which is the appropriate JAX primitive for fixed iterations. The `body_fn` signature is changed to match `fori_loop`'s requirements (taking index and carry state), and the unused parameters `tol` and `max_iter` could be removed or used if needed. This allows JAX to properly compile and optimize the iterative computation.

### 27. qe-jax-007 — Use consistent function naming for updates
**Location:** Line 454 / Section "Using jax.lax.scan for efficient iteration"
**Description:** The function `update_distribution` does not follow the `[quantity]_update` naming pattern and lacks the required time step parameter for consistency.

```
```{code-cell} ipython3
def update_distribution(psi, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(psi)):
        result.append(jnp.dot(P[:, i], psi))
    return jnp.array(result)
```

**Suggestion:**

```
```{code-cell} ipython3
def distribution_update(current_distribution, time_step, P):
    """Update the distribution by one step."""
    result = []
    for i in range(len(current_distribution)):
        result.append(jnp.dot(P[:, i], current_distribution))
    return jnp.array(result)
```

**Explanation:** This fix follows the `[quantity]_update` naming convention where "distribution" is the quantity being updated, and includes the `time_step` parameter for consistency with the style rule, even though it's not used in this implementation. The parameter name is also changed from `psi` to `current_distribution` to be more descriptive.

### 28. qe-jax-004 — Use functional update patterns
**Location:** Line 147 / Section "Rolling our own"
**Description:** Uses in-place array assignment `X[0] = init` instead of JAX functional update pattern.

```
def mc_sample_path(P, init=0, sample_size=1_000):
    """Simulate a Markov chain sample path."""
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)
    X[0] = init
    n = len(P)
```

**Suggestion:**

```
def mc_sample_path(P, init=0, sample_size=1_000):
    """Simulate a Markov chain sample path."""
    P = np.asarray(P)
    X = np.empty(sample_size, dtype=int)
    X = X.at[0].set(init)
    n = len(P)
```

**Explanation:** Replaces in-place assignment with JAX's functional `.at[].set()` pattern, which returns a new array rather than mutating the original.

---

### 29. qe-jax-004 — Use functional update patterns
**Location:** Line 151 / Section "Rolling our own"
**Description:** Uses in-place array assignment `X[t+1] = ...` in a loop instead of JAX functional update pattern.

```
for t in range(sample_size - 1):
        X[t+1] = qe.random.draw(P_dist[X[t]])
    return X
```

**Suggestion:**

```
for t in range(sample_size - 1):
        X = X.at[t+1].set(qe.random.draw(P_dist[X[t]]))
    return X
```

**Explanation:** Replaces in-place indexed assignment with JAX's functional `.at[].set()` pattern to avoid array mutation.

---

### 30. qe-jax-004 — Use functional update patterns
**Location:** Line 610 / Section "Exercise solution mc-jax-ex1"
**Description:** Uses in-place array assignment `states[0] = 0` instead of JAX functional update pattern.

```
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states[0] = 0

P_np = np.array(P_growth)
```

**Suggestion:**

```
n_periods = 10_000
states = np.empty(n_periods, dtype=int)
states = states.at[0].set(0)

P_np = np.array(P_growth)
```

**Explanation:** Uses JAX functional update pattern `.at[].set()` instead of direct in-place assignment for consistency with JAX idioms.

---

### 31. qe-jax-004 — Use functional update patterns
**Location:** Line 614 / Section "Exercise solution mc-jax-ex1"
**Description:** Uses in-place array assignment `states[t+1] = ...` in a loop instead of JAX functional update pattern.

```
P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states[t+1] = np.random.choice(3, p=P_np[states[t]])

for i, label in enumerate(["Recession", "Normal", "Boom"]):
```

**Suggestion:**

```
P_np = np.array(P_growth)
for t in range(n_periods - 1):
    states = states.at[t+1].set(np.random.choice(3, p=P_np[states[t]]))

for i, label in enumerate(["Recession", "Normal", "Boom"]):
```

**Explanation:** Replaces in-place indexed assignment with JAX's `.at[].set()` functional pattern to avoid array mutation and align with JAX best practices.

### 32. qe-jax-006 — Explicit PRNG key management
**Location:** Line 166 / Section "Computing with JAX"
**Description:** Uses NumPy's implicit random state (`np.random.seed()` and `np.random.dirichlet`) to generate data that will be used with JAX arrays, instead of using JAX's explicit PRNG key management.

```
# Build a larger transition matrix
n = 50
np.random.seed(42)
P_large = np.random.dirichlet(np.ones(n), size=n)
P_large = jnp.array(P_large)
```

**Suggestion:**

```
# Build a larger transition matrix
n = 50
import jax.random as jr
key = jr.PRNGKey(42)
P_large = jr.dirichlet(key, np.ones(n), shape=(n,))
```

**Explanation:** This fix replaces NumPy's implicit random state with JAX's explicit PRNG key management using `jr.PRNGKey(42)` and `jr.dirichlet()`, eliminating the need for `np.random.seed()` and directly producing a JAX array without conversion.

### 33. qe-jax-006 — Explicit PRNG key management
**Location:** Line 398 / Section "Exercise mc-jax-ex1 solution"
**Description:** Uses NumPy's `np.random.seed()` and `np.random.choice()` for simulation despite importing `jax.random` and creating a JAX PRNG key that is never used. The exercise explicitly asks to "Use JAX for the computation."

```
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

**Explanation:** This fix removes the NumPy random seed and uses the JAX PRNG key that was already created. It properly splits the key at each iteration using `jr.split()` and uses `jr.choice()` instead of `np.random.choice()`, following JAX's explicit key management pattern.

### 34. qe-fig-001 — Do not set figure size unless necessary
**Location:** Line 168 / Section "Convergence to stationarity"
**Description:** The code explicitly sets `figsize=(10, 6)` when creating a matplotlib figure without providing justification. QuantEcon lecture series set figure size defaults in `_config.yml` that should be used unless there is a specific reason to override them.

```
fig, ax = plt.subplots(figsize=(10, 6))
```

**Suggestion:**

```
fig, ax = plt.subplots()
```

**Explanation:** Removing the `figsize` parameter allows the figure to use the default size configured in `_config.yml`, maintaining consistency across the QuantEcon lecture series. If a custom size is truly needed for this particular visualization, it should be accompanied by a comment explaining why.

### 35. qe-fig-002 — Prefer code-generated figures
**Location:** Line 244-249 / Section "Stationary distributions"
**Description:** A static PNG image file is referenced via a `{figure}` directive when the identical plot is already generated by the code cell immediately preceding it. The code uses matplotlib to create and display the convergence plot, making the static image redundant.

```
```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: fig-convergence-plot
:width: 80%

Convergence to Hamilton's stationary distribution
```

**Suggestion:**

```
The figure above shows how the distribution converges over time.

The initial condition doesn't matter for the long-run outcome.
```

**Explanation:** The static figure directive should be removed entirely because the code cell immediately above (lines 221-242) already generates and displays this plot using matplotlib. The code-generated figure will automatically appear as the output of that code cell in the rendered notebook, making the static PNG file unnecessary. References to the figure by number should be replaced with descriptive text since the plot is now part of the code cell output rather than a separately numbered figure.

---

## Warnings (1)

- ⚠️  Could not apply qe-code-003: Text changed since parsing

---

## ✅ Applied Fixes (58)

> **No action required:** The following rule violations were automatically fixed in the lecture file.

### 1. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 449-450 / Section "Exercises"
**Description:** This paragraph block contains two sentences without blank line separation.
**Current text:**

```
Write a function using `jax.lax.fori_loop` that computes $\psi P^t$ for a given initial distribution $\psi$ and transition matrix $P$, returning the distribution at each time step. Compare the performance with a pure Python loop.
```

**Applied fix:**

```
Write a function using `jax.lax.fori_loop` that computes $\psi P^t$ for a given initial distribution $\psi$ and transition matrix $P$, returning the distribution at each time step.

Compare the performance with a pure Python loop.
```

**Explanation:** The two sentences are now separated by a blank line, creating two paragraph blocks.

### 2. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 387-388 / Section "A Larger Economic Model"
**Description:** This paragraph block contains two sentences without blank line separation.
**Current text:**

```
Consider a model where workers transition between income quartiles. The model uses a $4 \times 4$ transition matrix.
```

**Applied fix:**

```
Consider a model where workers transition between income quartiles.

The model uses a $4 \times 4$ transition matrix.
```

**Explanation:** The two sentences are now separated by a blank line, creating two paragraph blocks.

### 3. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 264-265 / Section "Computing with JAX"
**Description:** This paragraph block contains two sentences without blank line separation.
**Current text:**

```
JAX provides significant performance advantages for iterative computations. Let's use it to compute stationary distributions via the Power Method.
```

**Applied fix:**

```
JAX provides significant performance advantages for iterative computations.

Let's use it to compute stationary distributions via the Power Method.
```

**Explanation:** The two sentences are now separated by a blank line, creating two paragraph blocks.

### 4. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 233-234 / Section "Convergence to Stationarity"
**Description:** This paragraph block contains two sentences without blank line separation.
**Current text:**

```
Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition. This is a powerful result.
```

**Applied fix:**

```
Part 2 of the convergence theorem tells us that the marginal distribution of $X_t$ converges to $\psi^*$ regardless of the initial condition.

This is a powerful result.
```

**Explanation:** The two sentences are now separated by a blank line, creating two paragraph blocks.

### 5. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 118-119 / Section "An Employment Model"
**Description:** This paragraph block contains two sentences without blank line separation.
**Current text:**

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run. These are basically the questions we want to answer in this lecture.
```

**Applied fix:**

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.

These are basically the questions we want to answer in this lecture.
```

**Explanation:** The two sentences are now separated by a blank line, creating two paragraph blocks.

### 6. qe-writing-001 — Use one sentence per paragraph
**Location:** Line 22-24 / Section "Overview"
**Description:** This paragraph block contains three sentences without blank line separation between them.
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

**Explanation:** Each sentence is now in its own paragraph block, separated by blank lines.

### 7. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 357 (approx) / Section "Exercises"
**Description:** Common noun phrase "Economic Growth" is unnecessarily capitalized mid-sentence.
**Current text:**

```
Consider the following model of Economic Growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).
```

**Applied fix:**

```
Consider the following model of economic growth where a country can be in one of three states: recession (state 0), normal growth (state 1), or boom (state 2).
```

**Explanation:** "Economic growth" is a common noun phrase describing a concept, not a proper noun.

### 8. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 339 (approx) / Section "A Larger Economic Model"
**Description:** Technical term "Rate of Convergence" is unnecessarily capitalized mid-sentence.
**Current text:**

```
This spectral decomposition helps us understand the Rate of Convergence to the stationary distribution.
```

**Applied fix:**

```
This spectral decomposition helps us understand the rate of convergence to the stationary distribution.
```

**Explanation:** "Rate of convergence" is a technical term but not a proper noun.

### 9. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 216 (approx) / Section "Convergence to Stationarity"
**Description:** Common nouns "Marginal Distributions" and "Stationary Distribution" and "Recession Model" are unnecessarily capitalized in figure caption.
**Current text:**

```
Convergence of Marginal Distributions to the Stationary Distribution for Hamilton's Recession Model
```

**Applied fix:**

```
Convergence of marginal distributions to the stationary distribution for Hamilton's recession model
```

**Explanation:** Figure captions are narrative text. Only "Hamilton's" is a proper noun; "marginal distributions", "stationary distribution", and "recession model" are common nouns.

### 10. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 173 (approx) / Section "Computing the Stationary Distribution"
**Description:** Technical term "Stationary Distribution" is unnecessarily capitalized mid-sentence.
**Current text:**

```
For our employment model, we can find the Stationary Distribution analytically.
```

**Applied fix:**

```
For our employment model, we can find the stationary distribution analytically.
```

**Explanation:** "Stationary distribution" is a technical term but not a proper noun.

### 11. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 159 (approx) / Section "Theory"
**Description:** Technical term "Stochastic Steady States" is unnecessarily capitalized.
**Current text:**

```
Hence stationary distributions represent **Stochastic Steady States**.
```

**Applied fix:**

```
Hence stationary distributions represent **stochastic steady states**.
```

**Explanation:** "Stochastic steady states" is a technical term but not a proper noun.

### 12. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 56 (approx) / Section "An Employment Model"
**Description:** Common noun "Transition Matrix" is unnecessarily capitalized mid-sentence.
**Current text:**

```
The Transition Matrix is
```

**Applied fix:**

```
The transition matrix is
```

**Explanation:** "Transition matrix" is a technical term but not a proper noun, so it should be lowercase.

### 13. qe-writing-004 — Avoid unnecessary capitalization in narrative text
**Location:** Line 36 (approx) / Section "The Markov Property"
**Description:** Common noun "Current State" is unnecessarily capitalized mid-sentence.
**Current text:**

```
In other words, knowing the Current State is enough to determine probabilities for future states.
```

**Applied fix:**

```
In other words, knowing the current state is enough to determine probabilities for future states.
```

**Explanation:** "Current state" is a common noun phrase, not a proper noun, and should not be capitalized mid-sentence.

### 14. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 239 / Section "A Larger Economic Model"
**Description:** Section heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
## A Larger Economic Model
```

**Applied fix:**

```
## A larger economic model
```

**Explanation:** Section headings (H2) should only capitalize the first word and proper nouns. "Larger," "Economic," and "Model" are not proper nouns.

### 15. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 266 / Subsection "Using jax.lax.scan for Efficient Iteration"
**Description:** Subsection heading capitalizes words beyond the first word that are not proper nouns.
**Current text:**

```
### Using jax.lax.scan for Efficient Iteration
```

**Applied fix:**

```
### Using jax.lax.scan for efficient iteration
```

**Explanation:** Subsection headings (H3) should only capitalize the first word and proper nouns. "Efficient" and "Iteration" are not proper nouns and should be lowercase.

### 16. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 218 / Subsection "The Power Method"
**Description:** Subsection heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
### The Power Method
```

**Applied fix:**

```
### The power method
```

**Explanation:** Subsection headings (H3) should only capitalize the first word and proper nouns. "Power" and "Method" are not proper nouns.

### 17. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 160 / Subsection "Convergence to Stationarity"
**Description:** Subsection heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
### Convergence to Stationarity
```

**Applied fix:**

```
### Convergence to stationarity
```

**Explanation:** Subsection headings (H3) should only capitalize the first word and proper nouns. "Stationarity" is not a proper noun.

### 18. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 142 / Subsection "Computing the Stationary Distribution"
**Description:** Subsection heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
### Computing the Stationary Distribution
```

**Applied fix:**

```
### Computing the stationary distribution
```

**Explanation:** Subsection headings (H3) should only capitalize the first word and proper nouns. "Stationary" and "Distribution" are not proper nouns.

### 19. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 120 / Section "Stationary Distributions"
**Description:** Section heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
## Stationary Distributions
```

**Applied fix:**

```
## Stationary distributions
```

**Explanation:** Section headings (H2) should only capitalize the first word and proper nouns. "Distributions" is not a proper noun.

### 20. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 78 / Section "Simulating Markov Chains"
**Description:** Section heading capitalizes the word "Chains" when only proper nouns beyond the first word should be capitalized.
**Current text:**

```
## Simulating Markov Chains
```

**Applied fix:**

```
## Simulating Markov chains
```

**Explanation:** While "Markov" is a proper noun and should be capitalized, "Chains" is not and should be lowercase.

### 21. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 50 / Section "An Employment Model"
**Description:** Section heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
## An Employment Model
```

**Applied fix:**

```
## An employment model
```

**Explanation:** Section headings (H2) should only capitalize the first word and proper nouns. "Employment" and "Model" are not proper nouns.

### 22. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 44 / Subsection "The Markov Property"
**Description:** Subsection heading capitalizes the word "Property" when only proper nouns beyond the first word should be capitalized.
**Current text:**

```
### The Markov Property
```

**Applied fix:**

```
### The Markov property
```

**Explanation:** While "Markov" is a proper noun and should be capitalized, "Property" is not and should be lowercase.

### 23. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 34 / Subsection "Stochastic Matrices"
**Description:** Subsection heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
### Stochastic Matrices
```

**Applied fix:**

```
### Stochastic matrices
```

**Explanation:** Subsection headings (H3) should only capitalize the first word and proper nouns. "Matrices" is not a proper noun.

### 24. qe-writing-006 — Capitalize lecture titles properly
**Location:** Line 32 / Section "Definitions and Setup"
**Description:** Section heading capitalizes all major words instead of only the first word and proper nouns.
**Current text:**

```
## Definitions and Setup
```

**Applied fix:**

```
## Definitions and setup
```

**Explanation:** Section headings (H2) should only capitalize the first word and proper nouns. "Setup" is not a proper noun.

### 25. qe-writing-005 — Use bold for definitions, italic for emphasis
**Location:** Line 103 / Section "Stationary distributions"
**Description:** The phrase "stochastic steady states" is bolded for emphasis/interpretation rather than as a formal definition. The formal definition was already given (stationary distribution). The word "Hence" and the context ("represent") indicate this is emphasizing an interpretation rather than defining a new term, so italic formatting should be used instead of bold.
**Current text:**

```
Hence stationary distributions represent **stochastic steady states**.
```

**Applied fix:**

```
Hence stationary distributions represent *stochastic steady states*.
```

**Explanation:** This change replaces bold with italic formatting because the text is emphasizing an interpretive concept rather than formally defining a new term. The stationary distribution was already defined; this sentence emphasizes what it represents in economic terms, which is emphasis rather than definition.

### 26. qe-math-001 — Prefer UTF-8 unicode for simple parameter mentions, be consistent
**Location:** Line 88 / Section "An employment model"
**Description:** The parameters α and β are mentioned in narrative text using inline math delimiters (`$\alpha$` and `$\beta$`), but this sentence contains no mathematical expressions. According to the rule, simple parameter mentions without associated mathematical expressions should use UTF-8 unicode characters for better readability.
**Current text:**

```
Once we have the values of $\alpha$ and $\beta$, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.
```

**Applied fix:**

```
Once we have the values of α and β, we can address questions like what is the average duration of unemployment, and what fraction of time does a worker spend unemployed in the long run.
```

**Explanation:** This sentence is simple narrative text mentioning parameter values without any mathematical expressions or operators. Using UTF-8 unicode characters (α, β) instead of inline math reduces visual clutter and improves readability, which is the preferred style according to qe-math-001.

### 27. qe-math-002 — Use \top for transpose notation
**Location:** Line 391 / Section "A larger economic model"
**Description:** Inline math for matrix transpose uses `^T` instead of `^\top`.
**Current text:**

```
The eigenvalues of $P^T$ determine how quickly the distribution converges.
```

**Applied fix:**

```
The eigenvalues of $P^\top$ determine how quickly the distribution converges.
```

**Explanation:** Replaces `^T` with `^\top` in the inline math expression to follow the required transpose notation convention.

### 28. qe-math-002 — Use \top for transpose notation
**Location:** Line 383 / Section "A larger economic model"
**Description:** The spectral decomposition formula uses `^T` for transpose notation instead of `^\top`.
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

**Explanation:** Replaces `^T` with `^\top` to comply with the standard transpose notation rule.

### 29. qe-math-003 — Use square brackets for matrix notation
**Location:** Line 80 / Section "An employment model"
**Description:** The transition matrix P uses `\begin{pmatrix}` (parentheses) instead of the required `\begin{bmatrix}` (square brackets).
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

**Explanation:** Replacing `\begin{pmatrix}...\end{pmatrix}` with `\begin{bmatrix}...\end{bmatrix}` ensures the matrix uses square brackets as required by the QuantEcon style guide.

### 30. qe-math-004 — Do not use bold face for matrices or vectors
**Location:** Section "A larger economic model" (after the code cell printing stationary distribution)
**Description:** The passage uses `\mathbf{P}`, `\mathbf{L}`, `\mathbf{D}` for matrices, violating the rule against bold face formatting for matrices and vectors.
**Current text:**

```
The income transition matrix $\mathbf{P}$ can be decomposed as

$$
\mathbf{P} = \mathbf{L} \mathbf{D} \mathbf{L}^\top
$$

where $\mathbf{L}$ is a lower triangular matrix and $\mathbf{D}$ is diagonal.
```

**Applied fix:**

```
The income transition matrix $P$ can be decomposed as

$$
P = L D L^\top
$$

where $L$ is a lower triangular matrix and $D$ is diagonal.
```

**Explanation:** This fix removes all `\mathbf` commands and replaces them with plain letter notation for the matrices $P$, $L$, and $D$, conforming to the style rule that prohibits bold face formatting for matrices and vectors.

### 31. qe-math-007 — Use automatic equation numbering, not manual tags
**Location:** Line 142 / Section "Stationary distributions"
**Description:** The equation defining stationary distributions uses `\tag{1}` for manual equation numbering instead of the automatic numbering system with labels.
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

**Explanation:** This fix removes the manual `\tag{1}` and replaces it with MyST's automatic equation numbering system using a label `(stationary-dist)`. The equation can then be referenced elsewhere using `{eq}`stationary-dist`` if needed.

### 32. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 405-420 / Section "Exercise solution 1"
**Description:** Variable `psi` is spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
# Compute stationary distribution
psi = jnp.ones(3) / 3
for _ in range(1000):
    psi = psi @ P_growth

print(f"Stationary distribution: {psi}")
```

**Applied fix:**

```
# Compute stationary distribution
ψ = jnp.ones(3) / 3
for _ in range(1000):
    ψ = ψ @ P_growth

print(f"Stationary distribution: {ψ}")
```

**Explanation:** Using Unicode `ψ` for the distribution variable aligns with the standard notation used throughout the lecture.

### 33. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 340-355 / Section "A larger economic model"
**Description:** Variable `psi` is spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
# Convert to JAX and compute stationary distribution
income_P_jax = jnp.array(income_P)
psi = jnp.ones(4) / 4
for _ in range(500):
    psi = psi @ income_P_jax

print("Stationary distribution of income quartiles:")
quartile_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
for label, prob in zip(quartile_labels, psi):
    print(f"  {label}: {prob:.4f}")
```

**Applied fix:**

```
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

**Explanation:** Using Unicode `ψ` maintains consistency with the mathematical notation used for distributions.

### 34. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 308-320 / Section "Using jax.lax.scan for efficient iteration"
**Description:** Variables `psi`, `psi_0`, `psi_new`, `psi_final`, and `psi_history` are spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
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

**Applied fix:**

```
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

**Explanation:** Consistent use of Unicode `ψ` throughout the function aligns with mathematical notation.

### 35. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 270-280 / Section "JIT-compiled version"
**Description:** Variables `psi` and `psi_new` are spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
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

**Applied fix:**

```
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

**Explanation:** Using Unicode `ψ` improves code readability and matches mathematical convention.

### 36. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 252 / Section "The power method"
**Description:** Variable `psi_star` is spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
solver = StationarySolver(P_large)
psi_star, iterations = solver.solve()
print(f"Converged in {iterations} iterations")
print(f"Distribution sums to: {jnp.sum(psi_star):.10f}")
```

**Applied fix:**

```
solver = StationarySolver(P_large)
ψ_star, iterations = solver.solve()
print(f"Converged in {iterations} iterations")
print(f"Distribution sums to: {jnp.sum(ψ_star):.10f}")
```

**Explanation:** Using Unicode `ψ` maintains consistency with mathematical notation throughout the document.

### 37. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 230-245 / Section "The power method"
**Description:** Variables `psi` and `psi_new` are spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
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
def solve(self):
        ψ = jnp.ones(self.n) / self.n
        for i in range(self.max_iter):
            ψ_new = ψ @ self.P
            if jnp.max(jnp.abs(ψ_new - ψ)) < self.tol:
                return ψ_new, i + 1
            ψ = ψ_new
        return ψ, self.max_iter
```

**Explanation:** Using Unicode `ψ` makes the variable naming consistent with mathematical notation for distributions.

### 38. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 179-195 / Section "Convergence to stationarity"
**Description:** Variable `psi` is spelled out instead of using Unicode symbol `ψ`.
**Current text:**

```
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
```

**Applied fix:**

```
ψ = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Convergence to Stationary Distribution")
ax.set_xlabel("Iteration")
ax.set_ylabel("Probability")

ψ_history = [ψ.copy()]
for t in range(30):
    ψ = ψ @ P_hamilton
    ψ_history.append(ψ.copy())

ψ_history = np.array(ψ_history)
```

**Explanation:** Using Unicode `ψ` for the distribution variable aligns with standard mathematical notation.

### 39. qe-code-002 — Use Unicode symbols for Greek letters in code
**Location:** Line 86-90 / Section "An employment model"
**Description:** Variables `alpha` and `beta` are spelled out instead of using Unicode symbols `α` and `β`.
**Current text:**

```
alpha = 0.3    # probability of finding a job
beta = 0.2     # probability of losing a job

P = np.array([[1 - alpha, alpha],
              [beta, 1 - beta]])
```

**Applied fix:**

```
α = 0.3    # probability of finding a job
β = 0.2     # probability of losing a job

P = np.array([[1 - α, α],
              [β, 1 - β]])
```

**Explanation:** Using Unicode `α` and `β` makes the code more readable and closer to the mathematical notation used in the lecture.

### 40. qe-code-003 — Package installation at lecture top
**Location:** Line 39-41 / Section "Overview"
**Description:** The installation code cell is missing the required `tags: [hide-output]` directive to suppress verbose installation output.
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

**Explanation:** Adding the `tags: [hide-output]` directive suppresses verbose pip installation output, keeping the rendered lecture clean and focused on the content.

### 41. qe-jax-002 — Use NamedTuple for model parameters
**Location:** Line 267 / Section "### The power method"
**Description:** The `StationarySolver` class uses a mutable class with `__init__` to store solver parameters (P, tol, max_iter, n) without a factory function for validation. This violates the rule requiring NamedTuple for parameter storage with factory functions.
**Current text:**

```
class StationarySolver:
    """Solver for stationary distributions using the power method."""
    
    def __init__(self, P, tol=1e-10, max_iter=10_000):
        self.P = P
        self.tol = tol
        self.max_iter = max_iter
        self.n = P.shape[0]
    
    def solve(self):
        ψ = jnp.ones(self.n) / self.n
        for i in range(self.max_iter):
            ψ_new = ψ @ self.P
            if jnp.max(jnp.abs(ψ_new - ψ)) < self.tol:
                return ψ_new, i + 1
            ψ = ψ_new
        return ψ, self.max_iter
```

**Applied fix:**

```
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

**Explanation:** This fix replaces the mutable class with an immutable NamedTuple (`StationaryParams`) for parameter storage and provides a factory function (`create_stationary_params`) with validation logic. The solve logic is moved to a standalone function that accepts the parameters. This makes the parameters immutable, adds validation, and follows the functional programming style preferred in JAX code.

### 42. qe-fig-003 — No matplotlib embedded titles
**Location:** Line 203 / Section "Convergence to stationarity"
**Description:** The code uses `ax.set_title()` to embed a title directly in the matplotlib figure. This violates the rule as it is not within an exercise or solution directive, and titles should be added using mystnb metadata or figure directive instead.
**Current text:**

```
```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

ψ = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Convergence to Stationary Distribution")
ax.set_xlabel("Iteration")
ax.set_ylabel("Probability")

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

**Applied fix:**

```
```{code-cell} ipython3
P_hamilton = np.array([
    [0.971, 0.029, 0.000],
    [0.145, 0.778, 0.077],
    [0.000, 0.508, 0.492]
])

ψ = np.array([0.0, 0.2, 0.8])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Iteration")
ax.set_ylabel("Probability")

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

**Explanation:** Removed the `ax.set_title("Convergence to Stationary Distribution")` line. The title should instead be added using mystnb cell metadata (e.g., `mystnb: figure: name: "Convergence to Stationary Distribution"`) or through a figure directive wrapper if this output is to be referenced elsewhere in the document.

### 43. qe-fig-004 — Caption formatting conventions
**Location:** Line 302 / Section "Convergence to stationarity"
**Description:** The caption exceeds the maximum length of 5-6 words, containing 12 words total. The caption is verbose and should be more concise.
**Current text:**

```
Convergence of marginal distributions to the stationary distribution for Hamilton's recession model
```

**Applied fix:**

```
Convergence to Hamilton's stationary distribution
```

**Explanation:** This fix reduces the caption from 12 words to 5 words while maintaining the key information (convergence, Hamilton's model, and stationary distribution). The caption follows proper capitalization (first letter capitalized, proper noun "Hamilton's" capitalized, rest lowercase) and is concise per the rule requirements.

### 44. qe-fig-005 — Descriptive figure names for cross-referencing
**Location:** Line 241 / Section "Convergence to stationarity"
**Description:** The figure name `convergence-plot` does not follow the required `fig-` prefix convention. All figure names must use the pattern `fig-description` with lowercase and hyphens.
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

**Explanation:** Adding the `fig-` prefix makes the name consistent with the required naming convention `fig-description`. The corresponding cross-reference on line 247 should also be updated from `{numref}`convergence-plot`` to `{numref}`fig-convergence-plot``.

### 45. qe-fig-006 — Lowercase axis labels
**Location:** Line 275 / Section "Convergence to stationarity"
**Description:** The y-axis label "Probability" uses Title Case capitalization instead of lowercase. "Probability" is not a proper noun and should be lowercase.
**Current text:**

```
ax.set_ylabel("Probability")
```

**Applied fix:**

```
ax.set_ylabel("probability")
```

**Explanation:** This changes the axis label to lowercase as required by the style guide, since "probability" is not a proper noun.

### 46. qe-fig-006 — Lowercase axis labels
**Location:** Line 274 / Section "Convergence to stationarity"
**Description:** The x-axis label "Iteration" uses Title Case capitalization instead of lowercase. "Iteration" is not a proper noun and should be lowercase.
**Current text:**

```
ax.set_xlabel("Iteration")
```

**Applied fix:**

```
ax.set_xlabel("iteration")
```

**Explanation:** This changes the axis label to lowercase as required by the style guide, since "iteration" is not a proper noun.

### 47. qe-fig-009 — Figure sizing
**Location:** Line 182 / Section "Convergence to stationarity"
**Description:** The figure directive does not specify an explicit width, making it impossible to ensure it falls within the recommended 80-100% text width range for optimal readability and layout.
**Current text:**

```
```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: fig-convergence-plot

Convergence to Hamilton's stationary distribution
```

**Applied fix:**

```
```{figure} /_static/lecture_specific/markov_chains/convergence_plot.png
:name: fig-convergence-plot
:width: 80%

Convergence to Hamilton's stationary distribution
```

**Explanation:** Adding an explicit `:width: 80%` directive ensures the figure is sized within the recommended 80-100% text width range, providing optimal readability and consistent layout across the document.

### 48. qe-ref-001 — Use correct citation style
**Location:** Line 544 / Section "Exercises"
**Description:** The author name "Hamilton" is manually written in the text followed by a standard `{cite}` command. This is manual citation formatting that should be replaced with an in-text citation using `{cite:t}` to properly integrate the author name into the sentence flow.
**Current text:**

```
The transition matrix is given by Hamilton {cite}`Hamilton2005`.
```

**Applied fix:**

```
The transition matrix is given by {cite:t}`Hamilton2005`.
```

**Explanation:** Using `{cite:t}` allows the citation system to properly format the author name as part of the sentence structure, avoiding manual duplication of the author name. This follows the correct pattern for in-text citations where the author name is part of the sentence flow.

### 49. qe-link-002 — Use doc links for cross-series references
**Location:** Line 358 / Section "A larger economic model"
**Description:** Markdown link with direct URL to the intro series (intro.quantecon.org) is used instead of a `{doc}` link with the `intro:` prefix.
**Current text:**

```
Also check [this lecture](https://intro.quantecon.org/markov_chains_I.html) for introductory material.
```

**Applied fix:**

```
Also check {doc}`this lecture<intro:markov_chains_I>` for introductory material.
```

**Explanation:** The markdown link points to a lecture in the intro series and should use the `{doc}` syntax with custom title format `{doc}`this lecture<intro:markov_chains_I>`` to maintain the same link text while using proper intersphinx referencing.

### 50. qe-link-002 — Use doc links for cross-series references
**Location:** Line 356 / Section "A larger economic model"
**Description:** Direct URL to the intermediate series (python.quantecon.org) is used instead of a `{doc}` link with the `intermediate:` prefix.
**Current text:**

```
For more details, see https://python.quantecon.org/wealth_dynamics.html on wealth distribution.
```

**Applied fix:**

```
For more details, see {doc}`intermediate:wealth_dynamics` on wealth distribution.
```

**Explanation:** The URL points to a lecture in the intermediate series (python.quantecon.org) and should use the `{doc}` syntax with the `intermediate:` prefix for proper cross-series referencing.

### 51. qe-admon-003 — Use tick count management for nested directives
**Location:** Line 577 / Section "Exercises"
**Description:** The solution-start directive contains a nested code-cell directive, but both use 3 backticks. The outer solution-start directive must use 4 backticks to properly contain the nested 3-backtick code-cell.
**Current text:**

```
```{solution-start} mc-jax-ex2
:class: dropdown
```

**Applied fix:**

```
````{solution-start} mc-jax-ex2
:class: dropdown
```

**Explanation:** Changed solution-start and solution-end to use 4 backticks, ensuring the outer directive has more ticks than the nested code-cell directive (3 ticks), following the standard nesting pattern.

### 52. qe-admon-003 — Use tick count management for nested directives
**Location:** Line 537 / Section "Exercises"
**Description:** The solution-start directive contains a nested code-cell directive, but both use 3 backticks. The outer solution-start directive must use 4 backticks to properly contain the nested 3-backtick code-cell.
**Current text:**

```
```{solution-start} mc-jax-ex1
:class: dropdown
```

**Applied fix:**

```
````{solution-start} mc-jax-ex1
:class: dropdown
```

**Explanation:** Changed solution-start and solution-end to use 4 backticks, ensuring the outer directive has more ticks than the nested code-cell directive (3 ticks), following the standard nesting pattern.

### 53. qe-admon-004 — Use prf prefix for proof directives
**Location:** Line 553 / Section "Exercises"
**Description:** The `solution-start` directive is missing the required `prf:` prefix. Solutions in sphinx-proof should use `prf:solution`.
**Current text:**

```
````{solution-start} mc-jax-ex2
:class: dropdown
```

**Applied fix:**

```
````{prf:solution} mc-jax-ex2
:class: dropdown
```

**Explanation:** The correct sphinx-proof syntax for solutions uses `prf:solution` directive instead of `solution-start`.

---

### 54. qe-admon-004 — Use prf prefix for proof directives
**Location:** Line 545 / Section "Exercises"
**Description:** The `exercise` directive is missing the required `prf:` prefix. All sphinx-proof directives must use the `prf:` namespace.
**Current text:**

```
```{exercise}
:label: mc-jax-ex2
```

**Applied fix:**

```
```{prf:exercise}
:label: mc-jax-ex2
```

**Explanation:** Adding the `prf:` prefix ensures the directive is properly recognized as a sphinx-proof exercise block.

---

### 55. qe-admon-004 — Use prf prefix for proof directives
**Location:** Line 543 / Section "Exercises"
**Description:** The `solution-end` directive is not proper sphinx-proof syntax. Solution blocks should be closed with standard directive closing.
**Current text:**

```
```{solution-end}
```

**Applied fix:**

```
````
```

**Explanation:** Sphinx-proof solution directives are closed with the standard MyST closing syntax, not with a `solution-end` directive.

---

### 56. qe-admon-004 — Use prf prefix for proof directives
**Location:** Line 577 / Section "Exercises"
**Description:** The `solution-end` directive is not proper sphinx-proof syntax. Solution blocks should be closed with standard directive closing.
**Current text:**

```
```{solution-end}
```

**Applied fix:**

```
````
```

**Explanation:** Sphinx-proof solution directives are closed with the standard MyST closing syntax, not with a `solution-end` directive.

### 57. qe-admon-004 — Use prf prefix for proof directives
**Location:** Line 505 / Section "Exercises"
**Description:** The `solution-start` directive is missing the required `prf:` prefix. Solutions in sphinx-proof should use `prf:solution`.
**Current text:**

```
````{solution-start} mc-jax-ex1
:class: dropdown
```

**Applied fix:**

```
````{prf:solution} mc-jax-ex1
:class: dropdown
```

**Explanation:** The correct sphinx-proof syntax for solutions uses `prf:solution` directive instead of `solution-start`.

---

### 58. qe-admon-004 — Use prf prefix for proof directives
**Location:** Line 493 / Section "Exercises"
**Description:** The `exercise` directive is missing the required `prf:` prefix. All sphinx-proof directives must use the `prf:` namespace.
**Current text:**

```
````{exercise}
:label: mc-jax-ex1
```

**Applied fix:**

```
````{prf:exercise}
:label: mc-jax-ex1
```

**Explanation:** Adding the `prf:` prefix ensures the directive is properly recognized as a sphinx-proof exercise block.

---
