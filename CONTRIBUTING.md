# Contributing

Thanks for your interest in improving F1 Telemetry Simulation & Lap Analysis Dashboard.

## Before making a change

1. Open an issue describing the bug, proposed feature or documentation change.
2. For larger changes, discuss the intended approach before starting work.
3. Keep changes focused and avoid unrelated formatting or refactoring.

## Development setup

Use Python 3.10 or newer and install the project dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Pull requests

- Explain what changed and why.
- Include steps to reproduce a bug or verify a feature.
- Update the README when commands, dependencies or user-facing behaviour change.
- Do not commit virtual environments, generated Excel exports, credentials or private telemetry.
- For GUI changes, include screenshots where practical.
- For simulation-model changes, explain the assumptions and avoid presenting heuristic outputs as validated vehicle data.
- Run a syntax check before submitting:

  ```bash
  python -m py_compile f1_live_telemetry.py
  ```

A syntax check is useful but does not replace testing the GUI and telemetry flow on a machine with the required desktop dependencies.
