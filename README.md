# NormalObjects Complaint Workflow

This project implements a LangGraph-based complaint workflow for NormalObjects. It classifies complaints, extracts intake details, validates category-specific issues, produces an investigation and resolution record, and closes the workflow with a structured summary.

The main runtime entrypoints are:

- `normalobjects.workflow.process_complaint(complaint: str)` for programmatic use
- `python -m normalobjects.demo` for the sample demo runner

## Requirements

- Python 3.12

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create a `.env` file with:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL` is optional. If omitted, the workflow defaults to `gpt-4o-mini`.

## Run

Run the demo workflow:

```bash
python -m normalobjects.demo
```

Run the test suite:

```bash
pytest -q
```

## Programmatic Use

```python
from normalobjects import workflow

result = workflow.process_complaint(
    "I am Nancy Wheeler. Today at 9pm, the Downside Up portal at Hawkins Lab opened three hours late."
)
```
