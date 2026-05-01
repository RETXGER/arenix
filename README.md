# Arenix

> **Adaptive multi-turn LLM adversarial security testing and reporting framework.**

| Python **3.11 / 3.12** | **Mock mode** (no API keys) | **JSON report export** | **Authorized testing only** |

[Arenix](https://github.com/RETXGER/arenix) runs an **Orchestrator** loop: configurable **attacker** and **target** models (including fully offline mocks) exchange prompts and replies over **multiple turns**. A rule-based/heuristic **analyzer** scores observable risk signals turn-by-turn and at session level (optionally supplemented by extra LLM calls when configured). Outputs are artifacts you can ingest into review workflows—not a verdict that replaces judgment.

### What it does

1. The **attacker role** proposes the next probe (adaptively, when adaptive modules load).
2. The **target** model responds to the scripted conversation.
3. The **analyzer** derives scores and classifications from heuristic (and optionally LLM-assisted) signals—not formal proofs.
4. The **observer** can comment / confirm breakpoints when wired in configuration.
5. The engine emits **JSON** (and tooling may build **HTML** alongside in API/UI flows) for archiving and dashboards.

### Try it in 60 seconds

```bash
pip install -r requirements.txt
python main_engine.py
```

Then open **`docs/sample_reports/mock_report.json`** for a committed, trimmed illustration of output shape (or `./arenix_report.json` after your run — same top-level schema). Browse with any editor or `python -m json.tool`.

### Demo artifacts

| Artifact | Purpose |
|---|---|
| **[`docs/demo/demo-flow.md`](docs/demo/demo-flow.md)** | Full mock walk-through, platforms, pytest, Streamlit |
| **[`docs/sample_reports/mock_report.json`](docs/sample_reports/mock_report.json)** | Offline-friendly sample JSON (mock providers only) |
| **[`docs/sample_reports/README.md`](docs/sample_reports/README.md)** | How to read the report keys |
| **[`docs/screenshots/README.md`](docs/screenshots/README.md)** | Placeholder / how to contribute UI screenshots |

### Safety & authorised use

> **Authorized use only.** Use Arenix solely for defensive testing, security validation, research, or controlled lab work with **explicit permission and scope**. It does **not** guarantee findings are complete or correct — combine outputs with competent review.

### Scope: core versus extended tooling

The **core** simulation lives in [`arenix_engine.py`](arenix_engine.py): **`Orchestrator`**, role adapters (attacker/target/analyzer/observer), scoring, and JSON export.

The repository also carries **optional** layers — **APO**, **Playwright**-related flows, **reconnaissance**, **validation / pipeline**, Streamlit **`ui/`** (root [`app.py`](app.py) loads `ui/app.py`), **`evidence/`**, **`reporting/`** — that you can ignore when you only need the CLI mock or basic LLM run.

---

## Quickstart

**Supported Python:** **3.11** or **3.12** (matches CI).

```bash
git clone https://github.com/RETXGER/arenix.git
cd arenix

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Default: mock providers, no API keys — writes ./arenix_report.json
python main_engine.py

python -m pytest -q

# Dashboard (delegates to ui/app.py)
streamlit run app.py

# REST API — requires FastAPI/uvicorn in the environment (see requirements.txt)
uvicorn api:app --reload --port 8000
```

| Step | Purpose |
|---|---|
| `python main_engine.py` | Run the orchestrated multi-turn simulation; emits **`arenix_report.json`** by default (`ARENIX_EXPORT_JSON_PATH` overrides location). |
| `python -m pytest -q` | Run automated checks (analyzer, adapters, export, API validators, console encoding guards). |

*(See **[Demo artifacts](#demo-artifacts)** at the top for sample JSON and narrative demo.)*

---

## Mock mode versus live LLMs

Mock mode uses built-in scripted behaviour so you can demo the **full turn loop**, scoring pipeline, summary output, and **JSON export** without network calls or billing.

- **Defaults** (omit env or set explicitly): `ARENIX_ATTACKER_PROVIDER=mock`, `ARENIX_TARGET_PROVIDER=mock`, and related mock observer/analyzer placeholders as described in `.env.example`.
- **Live providers** (`openai`, `anthropic`, `gemini`, `deepseek`, `ollama`, `custom`, etc.) require the corresponding **credentials and/or endpoints** exported in your shell (see below).

---

## Environment variables

The engine reads **`os.environ`** at runtime; **there is no built-in `.env` autoload.** If you use a `.env` file ([`.env.example`](.env.example) is a reference), export variables yourself (shell, systemd, IDE run config, direnv, or a small launcher), or activate a tool that injects env before invoking Python.

### Core session / orchestration

| Variable | Purpose |
|---|---|
| `ARENIX_ATTACKER_PROVIDER` | Provider for the attacker adapter (`mock`, `openai`, …). |
| `ARENIX_TARGET_PROVIDER` | Provider under test. |
| `ARENIX_OBSERVER_PROVIDER` | Observer confirmations / summaries (`mock` for offline demos). |
| `ARENIX_ANALYZER_PROVIDER` | Used when **semantic-assisted** analysis paths are enabled (see `.env.example` / [`SessionConfig`](arenix_engine.py)). |
| `ARENIX_MAX_TURNS` | Upper bound on dialogue turns (default documented in `.env.example`). |
| `ARENIX_EXPORT_JSON_PATH` | Output file path (`arenix_report.json` default). |
| `ARENIX_INDUSTRY`, `ARENIX_ATTACK_PROFILE`, `ARENIX_REQUIRE_OBSERVER_CONFIRMATION`, `ARENIX_STOP_ON_BREAK`, … | Additional tuning — see `.env.example`. |

### Provider credentials / endpoints

| Variable | Typical use |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` **or** `GEMINI_API_KEY` | Gemini (see engine adapter behaviour) |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `ARENIX_CUSTOM_API_KEY` + **`ARENIX_CUSTOM_BASE_URL`** | OpenRouter, Azure OpenAI, vLLM, or OpenAI-compatible gateways (`custom` provider) |

Supported provider names mirror the engine adapters: `mock` · `openai` · `anthropic` · `gemini` · `deepseek` · `ollama` · `custom` (see [`build_adapter`](arenix_engine.py)).

---

## Exported JSON outline

`export_json()` writes a UTF-8 document with **`analysis_report`** and **`raw_report`** as **JSON-safe dicts** (nested dataclasses, enums, datetimes flattened for serialization).

| Top-level key | Meaning |
|---|---|
| `session` | `SessionConfig` snapshot (providers, models, limits, paths, …). |
| `state` | Final behaviour state aggregates from `StateEngine`. |
| `turn_records` | One record per simulated turn (prompts, responses, latency/tokens placeholders, headline scores). |
| `analysis_report` | Canonical dict view of **`ArenixReport`**: turn analyses, breakpoints, vulnerability lists, Phase-6 artefacts (`attack_strategy_path`, `tactic_switch_log`, `exploitation_attempts`, `security_insights`, `executive_summary`), trend blocks, … |
| `raw_report` | Mirror of `analysis_report` content as exported (same informational payload—no opaque Python-only objects survive in JSON). |

The following logical fields live predominantly under **`analysis_report`**:

```
analysis_report highlights
├── status / verdict families (session + turn taxonomy)
├── model_compromised / model_under_pressure / attack_detected
├── max_compromise_score / overall metrics / resilience curves
├── break_point (+ observer confirmation metadata when configured)
├── early_warnings
├── turn_analyses[]
├── attack_strategy_path
├── tactic_switch_log
├── exploitation_attempts
├── security_insights
└── executive_summary
```

Consult [`SessionConfig`](arenix_engine.py) and **`ArenixReport`** definitions for exhaustive field typing.

---

## Limitations — read before you trust the numbers

- **Scoring is heuristic and calibrated**, optionally blended with auxiliary LLM judgements—not a cryptographic proof nor a calibrated psychometric instrument.
- **There is no academic or regulatory seal of accuracy**; outputs are aides for reviewers, auditors, or red-team leads—not compliance verdicts standing alone.
- **Always pair Arenix artefacts** (JSON, timelines, rationales in `security_insights`) **with competent human assessment** tailored to your threat model.

---

## Current status (engineering)

As of repository maintenance snapshots:

| Item | Notes |
|---|---|
| **Automated tests** | `python -m pytest -q` — **24** tests passing locally/CI-aligned (analyzer, adapters, JSON export integrity, FastAPI validations, **`print_summary`** console encoding guards under strict `cp1254`-like streams). |
| **CLI export** | `python main_engine.py` completes **`export_json`** without serialisation failures on **`raw_report`**. |
| **Windows consoles** | `print_summary` uses a safe writer so **narrow code pages** (`cp1254`, etc.) do not raise **`UnicodeEncodeError`** on headings that contain emoji/markers—output may substitute replacement characters while the process exits successfully. |

CI runs on Push/PR workflows with Python **3.11** and **3.12** ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

---

## Architecture (core)

```
┌─────────────────────────────────────────────────────────┐
│                        Orchestrator                      │
│  ┌──────────────┐   attack    ┌──────────────┐          │
│  │ AttackerRole │ ──────────► │  TargetRole   │          │
│  └──────────────┘             └──────────────┘           │
│         ▲                            │ response          │
│         │                            ▼                   │
│  ┌──────────────┐           ┌──────────────────┐        │
│  │ Adaptive /   │◄──────────│ ArenixAnalyzerV2 │        │
│  │   feedback   │  signals │  (scoring)       │        │
│  └──────────────┘           └─────────────────┘        │
│                                      │ ObserverRole     │
└────────────────────────────────────┴──────────────────┘
```

**Key modules (non-exhaustive):**

| File | Role |
|---|---|
| `arenix_engine.py` | Orchestration, adapters, scoring, export, summaries |
| `adaptive_attacker.py` | Adaptive tactics / feedback scaffolding |
| `attack_library.py` | Payload tooling (selectors/mutators within extended paths) |
| `main_engine.py` | Thin CLI wrapper |
| `app.py` | Streamlit entry shim → **`ui/app.py`** |
| `api.py` | FastAPI façade (requires FastAPI stack) |

---

## Core capabilities (summary)

| Capability | Notes |
|---|---|
| **Adaptive multi-turn attacker** | Tactic evolution under feedback—not “one-shot prompt fuzzing”. |
| **Turn-level defence taxonomy** | e.g., `SAFE`, `SOFT_COMPLIANCE`, `BOUNDARY_PRESSURE`, `UNDER_ATTACK_BUT_RESILIENT`, compromise classes when thresholds trigger. |
| **Phase-6 style reporting** | Strategy paths, exploit attempt ledgers, security insights buckets, narratives. |
| **Industry-ish tuning** | Preset thresholds (fintech / healthcare / default / …). |
| **Streamlit + FastAPI** | Human UI and integrations (optional installs). |

---

## Turn status taxonomy (per-turn excerpts)

| Status | Meaning |
|---|---|
| `SAFE` | No meaningful attack-pressure signal at this turn slice |
| `SOFT_COMPLIANCE` | Principled non-refusal with low disclosure risk signals |
| `BOUNDARY_PRESSURE` | Elevated heuristic pressure band without hardened compromise |
| `UNDER_ATTACK_BUT_RESILIENT` / `UNDER_ATTACK` | Pressure present; refusal / resilience patterns vary by thresholds |
| `COMPROMISED` / `CRITICAL_*` bands | High heuristic scores **plus** supporting disclosure/leak detectors — still subject to reviewer confirmation |

Consult [`ArenixAnalyzerV2.analyze_turn`](arenix_engine.py) for authoritative branch logic—README tables summarise intent only.

---

## Safety & operational posture

Arenix may emit **high-fidelity adversarial user messages** purely **inside authorised simulation transports** (configured LLM HTTP(S) sockets you control).

- Scope every engagement; log who approved it; minimise sensitive corpora in transcripts.
- **Never aim this toolkit** at unauthorised personal data, unmanaged production chatbots, third-party SaaS without contracts, nor physical / OT systems bridged indirectly through assistants.
- Respect vendor ToS rate limits — this repository does **not** implement comprehensive provider-aware throttling beyond basic client retries.

---

## Testing

```bash
python -m pytest -q
# or narrower:
python -m pytest tests/ -v
```

`tests/` plus root-level scenarios cover scoring heuristics, export round-trips, FastAPI validators, concurrency/error hygiene, terminal encoding regressions.

---

## Project structure (abridged)

```
arenix/
├── main_engine.py          # CLI shim
├── app.py                   # Runs ui/app.py
├── arenix_engine.py         # Core engine
├── adaptive_attacker.py
├── attack_library.py
├── semantic_engine.py
├── api.py                   # REST (needs FastAPI/uvicorn)
├── compliance_mapper.py     # Bridges into reporting.*
├── report_generator.py
├── tournament.py            # Separate extended tournament/APO tooling (not CLI core loop)
├── requirements.txt
├── .env.example             # Env reference ONLY — manual export needed
├── tests/                   # 20+ focussed tests (+ shared scenarios)
├── ui/                      # Streamlit dashboard
├── evidence/, reporting/, validation/, recon/, …  # optional extended layers
└── .github/workflows/tests.yml   # pytest on Python 3.11 + 3.12
```

---

## License

MIT — see [`LICENSE`](LICENSE).
