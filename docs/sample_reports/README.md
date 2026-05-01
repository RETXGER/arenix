# Sample reports

- **`mock_report.json`** — Truncated mock-mode export (`mock` providers only). Synthetic `session_id`, shortened dialogue text for readability. Same top-level schema as [`arenix_report.json`](../../arenix_report.json) produced by `python main_engine.py`.

Regenerate from a full mock run then trim/anonymise if schema changes materially.

## How to read the report

Exports (full run or [`mock_report.json`](mock_report.json)) share the **same outer keys**. Use them top-down:

| Key | Start here |
|-----|-------------|
| **`session`** | What was configured: providers, models, `max_turns`, profile, paths (e.g. where JSON would be written). |
| **`state`** | Last-turn/high-level behavioural aggregates from the internal state engine after the simulation. |
| **`turn_records`** | One row per simulated turn — attacker prompt text, target answer, coarse scores — good for skim / CSV-style review. |
| **`analysis_report`** | Full structured **`ArenixReport`**: per-turn **`turn_analyses`**, trends, breakpoints, **`security_insights`**, **`executive_summary`**, Phase-6-style artefacts. |
| **`raw_report`** | JSON mirror of **`analysis_report`** in the exported file (no live Python objects). |

Dig into **`analysis_report.turn_analyses[]`** when you need per-turn signals (`attack_pressure_score`, refusal flags, classifications). **`session`** stays the quickest place to sanity-check that the run actually used mocks vs live backends.
