# Contributing Guide

Thank you for contributing to this project. Please follow these guidelines to keep the codebase consistent and maintain CI passing status.

---

## Development setup

```bash
git clone https://github.com/<your-username>/ml-pipeline-churn.git
cd ml-pipeline-churn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

---

## Code style

This project uses **black** for formatting and **flake8** for linting. Both run automatically on every PR via GitHub Actions.

Before committing:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=100
```

Key style rules:
- Maximum line length: 100 characters
- All public functions must have docstrings (Google style)
- All modules must have a module-level docstring explaining purpose, responsibilities, and usage

---

## Running tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

All PRs must maintain test coverage above 70%.

---

## Branching strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready code — protected, requires PR |
| `develop` | Integration branch for features |
| `feature/<name>` | New features or improvements |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation only |

---

## Pull request checklist

Before opening a PR, verify:

- [ ] `black` and `isort` have been run
- [ ] `flake8` passes with no errors
- [ ] All existing tests pass
- [ ] New code has corresponding tests
- [ ] Docstrings are present on all new functions and modules
- [ ] If a model is changed, the Model Card has been updated
- [ ] If training data changes, the Data Card has been updated
- [ ] Any new configuration values are added to the appropriate `configs/*.yaml` file

---

## Governance rules

- **Never commit raw data** — use `.gitignore` (already configured)
- **Never commit secrets** — use environment variables or a secrets manager
- **Model promotions require documentation** — add a description to the MLflow Registry entry before promoting to Production
- **Bias audits are mandatory after retraining** — run `src/models/evaluate.py` and review `evaluation_outputs/bias_audit.json`
