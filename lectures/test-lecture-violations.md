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

# Test Lecture: Style Violations

This lecture contains intentional style violations for testing the QuantEcon Style Checker.

Each section contains violations from a specific category.

**DO NOT FIX THESE VIOLATIONS** - They are intentional test cases.

## Math Violations

### Inline Math in Wrong Context

The equation $E[X] = \sum_{i=1}^{n} x_i p_i$ should be display math because it's important.

Also this long equation $\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$ is too complex for inline.

### Display Math Issues

Here is an equation without proper referencing:

$$
y = mx + b
$$

And here's display math that should probably be inline:

The slope m represents the rate of change.

### Inconsistent Notation

We use α for the **learning rate**, but later we'll call it η which is confusing.

The variable x represents output, but sometimes we use y for the same thing.

## Code Violations

### Missing Language Specifier

```python
def hello():
    print("Hello World")
```

### No Cell Tags

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# This code cell should have a :tags: directive
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

### Long Lines Without Wrapping

```{code-cell} ipython3
# This line is way too long and should be wrapped for better readability in the lecture materials
very_long_variable_name_that_makes_the_line_exceed_recommended_length = some_function_with_many_parameters(param1, param2, param3, param4, param5)
```

## Writing Violations

### Contractions

We don't use contractions in formal writing.

It's not appropriate and doesn't look professional.

You shouldn't use them either.

They won't be accepted in the style guide.

### Passive Voice

The model was trained on the dataset.

The results were analyzed by the researchers.

It was determined that the algorithm is effective.

The paper was written last year.

### Informal Language

So basically, this algorithm is pretty cool and works really well.

Anyway, let's move on to the next section which is kinda important.

### Second Person

You *should* understand this concept before proceeding.

If you don't get it, you can review the previous section.

When you implement this, make sure you test your code thoroughly.

## Figures Violations

### Missing Figure Reference

```{figure} /_static/lecture_specific/example/figure1.png
:name: fig-example-1

This is a figure caption.
```

The figure above shows the results.

But we never reference {numref}`fig-example-1` properly in the text flow.

### Caption Issues

```{figure} /_static/lecture_specific/example/figure2.png
:name: fig-example-2

fig 2
```

### No Alt Text

```{figure} /_static/lecture_specific/example/figure3.png
:name: fig-example-3

A detailed analysis of the time series data showing the correlation between variables X and Y over the study period from 2010 to 2023.
```

## Links Violations

### Bare URLs

Check out https://quantecon.org for more information.

Also see http://example.com/some/long/path/to/resource for details.

### Poor Link Text

Click [here](https://quantecon.org) to learn more.

For more information, see [this link](https://docs.python.org).

Read [this](https://numpy.org/doc/stable/) for numpy documentation.

### Broken Relative Links

See the [introduction](../intro.md) for background.

Also check [another lecture](./nonexistent-lecture.md) for related content.

## References Violations

### Inconsistent Citation Format

According to Smith (2020), the results are significant.

But {cite}`jones2019` disagrees.

As shown in [Brown, 2018], there's another perspective.

See also {cite:p}`wilson_2021`.

### Missing Bibliography Entry

The seminal work by {cite}`nonexistent_reference_2025` established the foundation.

### Informal Citations

As John Smith said in his famous 2019 paper, "this is important."
The Python documentation (python.org) explains this well.

## Admonitions Violations

### Overuse of Warnings

```{warning}
This is a warning about something minor.
```

```{warning}
Another warning that's not really that important.
```

```{warning}
Yet another warning - too many warnings dilute their impact.
```

### Wrong Admonition Type

```{note}
CRITICAL: The system will crash if you don't follow these steps exactly!
```

```{tip}
Warning: This could cause data loss if not handled properly.
```

### Missing Admonition Where Needed

**Important:** You must install the dependencies before running the code.

**Note:** This section requires advanced knowledge of linear algebra.

## JAX Violations

### Not Using JAX Idioms

```{code-cell} ipython3
import jax.numpy as jnp
from jax import jit

# Using regular Python loops instead of JAX vectorization
def slow_function(x):
    result = []
    for i in range(len(x)):
        result.append(x[i] ** 2)
    return jnp.array(result)
```

### Missing JIT Compilation

```{code-cell} ipython3
import jax.numpy as jnp

# This function should be JIT compiled for performance
def compute_intensive_function(x, y, z):
    return jnp.dot(x, y) + jnp.sum(z ** 2) + jnp.mean(x * y)
```

### Incorrect Array Creation

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np

# Mixing numpy and jax arrays inappropriately
x = np.array([1, 2, 3, 4, 5])  # Should use jnp.array for JAX code
y = jnp.array([1, 2, 3, 4, 5])
result = jnp.dot(x, y)  # Implicit conversion
```

## Summary

This lecture contains {math violations}, {code violations}, {writing violations}, and more.
All violations are intentional for testing purposes.

Don't fix them!
