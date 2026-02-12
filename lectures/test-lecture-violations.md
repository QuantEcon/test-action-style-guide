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

## Math violations

### Inline math in wrong context

[No change needed - this usage is acceptable per rule guidelines]

### Display math issues

Here is an equation without proper referencing:

$$
y = mx + b
$$

And here's display math that should probably be inline.

The slope m represents the rate of change.

### Inconsistent notation

We use α for the **learning rate**, but later we'll call it η.

This is confusing.

The variable x represents output, but sometimes we use y for the same thing.

## Code violations

### Missing language specifier

```
def hello():
    print("Hello World")
```

### No cell tags

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# This code cell should have a :tags: directive
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

### Long lines without wrapping

```{code-cell} ipython3
# This line is way too long and should be wrapped for better readability in the lecture materials
very_long_variable_name_that_makes_the_line_exceed_recommended_length = some_function_with_many_parameters(param1, param2, param3, param4, param5)
```

## Writing violations

### Contractions

We don't use contractions in formal writing.

It's not appropriate and doesn't look professional.

You shouldn't use them either.

They won't be accepted in the style guide.

### Passive voice

The model was trained on the dataset.

The results were analyzed by the researchers.

It was determined that the algorithm is effective.

The paper was written last year.

### Informal language

So basically, this algorithm is pretty cool and works really well.

Anyway, let's move on to the next section which is kinda important.

### Second person

You should understand this concept before proceeding. If you don't get it, you can review the previous section.
When you implement this, make sure you test your code thoroughly.

## Figures violations

### Missing figure reference

```{figure} /_static/lecture_specific/example/figure1.png
:name: fig-example-1

This is a figure caption.
```

The figure above shows the results. But we never reference {numref}`fig-example-1` *properly* in the text flow.

### Caption issues

```{figure} /_static/lecture_specific/example/figure2.png
:name: fig-example-2

fig 2
```

### No alt text

```{figure} /_static/lecture_specific/example/figure3.png
:name: fig-example-3

A detailed analysis of the time series data showing the correlation between variables X and Y over the study period from 2010 to 2023.
```

## Links violations

### Bare URLs

Check out https://quantecon.org for more information.
Also see http://example.com/some/long/path/to/resource for details.

### Poor link text

Click [here](https://quantecon.org) to learn more.
For more information, see [this link](https://docs.python.org).
Read [this](https://numpy.org/doc/stable/) for numpy documentation.

### Broken relative links

See the [introduction](../intro.md) for background.
Also check [another lecture](./nonexistent-lecture.md) for related content.

## References violations

### Inconsistent citation format

According to Smith (2020), the results are significant. But {cite}`jones2019` disagrees.
As shown in [Brown, 2018], there's another perspective. See also {cite:p}`wilson_2021`.

### Missing bibliography entry

The *seminal* work by {cite}`nonexistent_reference_2025` established the foundation.

### Informal citations

As John Smith said in his famous 2019 paper, "this is important."
The Python documentation (python.org) explains this well.

## Admonitions violations

### Overuse of warnings

```{warning}
This is a warning about something minor.
```

```{warning}
Another warning that's not really that important.
```

```{warning}
Yet another warning - too many warnings dilute their impact.
```

### Wrong admonition type

```{note}
CRITICAL: The system will crash if you don't follow these steps exactly!
```

```{tip}
Warning: This could cause data loss if not handled properly.
```

### Missing admonition where needed

*Important:* You must install the dependencies before running the code.

*Note:* This section requires advanced knowledge of linear algebra.

## JAX violations

### Not using JAX idioms

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

This lecture contains *math violations*, *code violations*, *writing violations*, and more.
All violations are intentional for testing purposes.

Don't fix them!
