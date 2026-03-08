# Contributing

## Setup

```bash
git clone https://github.com/SubhadipJana1409/day21-bulk-rnaseq-de
cd day21-bulk-rnaseq-de
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

```bash
black src/ tests/
ruff check src/ tests/
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Open a pull request with a clear description
