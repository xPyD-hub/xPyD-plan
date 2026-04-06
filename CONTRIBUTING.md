# Contributing

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Fork & branch from `main`
2. Write code + tests
3. Lint: `ruff check xpyd_plan tests`
4. Open a PR — CI must pass, two reviewer bots must approve

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new analyzer
fix: correct SLA threshold check
docs: update README
```

## Code Style

- Formatter/linter: `ruff`
- Type hints required for public APIs
- Tests required for new features
