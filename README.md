# test-action-style-guide

Test repository for [action-style-guide](https://github.com/QuantEcon/action-style-guide) - contains test lectures for validating the QuantEcon Style Checker GitHub Action.

## Purpose

This repository provides:
1. **Test lectures** with intentional style violations for regression testing
2. **Clean lectures** that should pass all checks (baseline validation)
3. **GitHub workflows** to test the action in a production-like environment

## Test Lectures

| File | Description |
|------|-------------|
| `lectures/test-lecture-violations.md` | Lecture with intentional violations across all categories |
| `lectures/test-lecture-clean.md` | Clean lecture that should pass all checks |

## Testing the Action

### Manual Trigger

The action can be triggered manually from the Actions tab to test specific categories:

1. Go to **Actions** → **Test Style Checker**
2. Click **Run workflow**
3. Select the lecture and categories to test

### Issue Comment Trigger

Create an issue with a comment like:
```
@github-actions style-guide lectures/test-lecture-violations.md --categories math,code
```

## Violation Categories

The test lectures contain violations in these categories:
- **math** - Display vs inline math, equation formatting
- **code** - Code block annotations, language specifiers
- **writing** - Contractions, passive voice, sentence structure
- **figures** - Figure references, captions, formatting
- **links** - URL formatting, link text quality
- **references** - Citation formatting, bibliography
- **admonitions** - Admonition usage and formatting
- **jax** - JAX-specific conventions

## Updating Test Content

When adding new rules to the style checker:
1. Add a corresponding violation to `test-lecture-violations.md`
2. Verify the action detects it
3. Update this README if needed

## Related

- [action-style-guide](https://github.com/QuantEcon/action-style-guide) - The main style checker action
- [lecture-python.myst](https://github.com/QuantEcon/lecture-python.myst) - Production lecture repository
