# Mock demo flow

This walkthrough shows that the **core engine** runs end-to-end with **mock** LLM adapters (no API keys, no outbound calls to real providers).

## Prerequisites

- **Python 3.11 or 3.12** (same as CI).
- Virtual environment recommended.

**Create venv (all platforms):**

```bash
python -m venv .venv
```

**Activate:**

```bash
# macOS / Linux (bash/zsh)
source .venv/bin/activate
```

```bat
REM Windows CMD
.venv\Scripts\activate.bat
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## Run the mock simulation

Default environment already selects mock providers for attacker, target, observer, and analyzer.

```bash
python main_engine.py
```

Optional: limit turns for a shorter run (still full pipeline):

**Linux / macOS (bash):**

```bash
ARENIX_MAX_TURNS=3 python main_engine.py
```

**Windows CMD:**

```bat
set ARENIX_MAX_TURNS=3
python main_engine.py
```

**Windows PowerShell:**

```powershell
$env:ARENIX_MAX_TURNS = "3"
python main_engine.py
```

## Expected console behaviour

- Log lines from `main_engine` / `ArenixEngine` describing sector, target provider, and max turns.
- A text **summary** block (status, scores, turn table, vulnerabilities, recommendations, trend).
- Process exits with code **0** on success.

## JSON output

- By default the engine writes **`arenix_report.json`** in the current working directory (override with `ARENIX_EXPORT_JSON_PATH`).
- The file must be **valid UTF-8 JSON** (parse with any JSON tool).
- Top-level keys (see also [README](../../README.md#exported-json-outline)):

| Key | Role |
|-----|------|
| `session` | Snapshot of run configuration (providers, models, `max_turns`, paths, …). |
| `state` | Final behaviour state from the internal state engine. |
| `turn_records` | Per-turn prompts, target replies, and headline metrics. |
| `analysis_report` | Full structured report (turn analyses, trends, Phase-6 style fields, …). |
| `raw_report` | Same semantic content as `analysis_report`, stored as a JSON object in the export file. |

## Committed sample artifact

A **trimmed, anonymised** mock export (single-turn illustration, synthetic session id, truncated messages) lives at:

**[`docs/sample_reports/mock_report.json`](../sample_reports/mock_report.json)**

It contains **no API keys** and **no real endpoints**—only mock provider labels and canned dialogue patterns from the built-in mock adapters.

## Automated checks

```bash
python -m pytest -q
```

This includes tests for analyser behaviour, adapter wiring, JSON export round-trips, API validation (when FastAPI is installed), and console encoding safety on narrow Windows code pages.

## Optional: Streamlit

```bash
streamlit run app.py
```

The root `app.py` delegates to `ui/app.py`. Extended UI features may import optional modules; the CLI mock path above does not require them.

## Screenshots

There is no dashboard screenshot checked into this repository by default. If you add one for documentation, place it under **`docs/screenshots/`** and reference it from the main README (see note in that folder).
