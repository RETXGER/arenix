# Arenix

**Adversarial AI Security Testing Platform**

Arenix is a professional-grade, multi-turn adversarial simulation framework for testing the security and robustness of large language models (LLMs). It pits an adaptive attacker agent against a target LLM over multiple turns, scores defensive behavior turn-by-turn, and produces structured security reports suitable for enterprise audits and compliance review.

> **Defensive Use Only.** Arenix is designed to identify LLM vulnerabilities in controlled, authorized testing environments. It is not intended for use against production systems without explicit authorization.

---

## Core Capabilities

| Capability | Description |
|---|---|
| **Adaptive Attack Engine** | Q-learning–guided attacker that switches tactics based on observed defenses |
| **7 Attack Dimensions** | Semantic poisoning, indirect injection, multi-turn escalation, role social engineering, context drift, jailbreak, data extraction |
| **Turn-Level Scoring** | Per-turn metrics: `compromise_score`, `attack_pressure`, `adaptive_resistance`, `semantic_drift`, `manipulation_probability` |
| **Defense Classification** | `SAFE` · `SOFT_COMPLIANCE` · `BOUNDARY_PRESSURE` · `UNDER_ATTACK_BUT_RESILIENT` · `COMPROMISED` · `CRITICAL_COMPROMISE` |
| **Session-Level Verdict** | `SAFE` · `UNDER_ATTACK` · `UNDER_PRESSURE` · `RESILIENT_UNDER_ATTACK` · `COMPROMISED` · `CRITICAL` |
| **Breakpoint Detection** | Detects first partial leakage, boundary weakening turn, and vulnerability type |
| **Phase 6 Reports** | `attack_strategy_path`, `tactic_switch_log`, `exploitation_attempts`, `security_insights`, `executive_summary` |
| **Industry Profiles** | Pre-calibrated thresholds for `fintech`, `healthcare`, `government`, `tech`, `default` |
| **Streamlit Dashboard** | Interactive UI with attack timeline visualization |
| **FastAPI Endpoint** | Programmatic access for CI/CD integration |
| **Compliance Mapping** | Optional OWASP LLM Top 10 and NIST CSF mapping |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Orchestrator                      │
│                                                         │
│  ┌──────────────┐   attack    ┌──────────────┐          │
│  │ AttackerRole │ ──────────► │  TargetRole  │          │
│  │ (adaptive)   │             │  (LLM under  │          │
│  └──────────────┘             │   test)      │          │
│         ▲                     └──────┬───────┘          │
│         │ counter-strategy           │ response         │
│         │                            ▼                  │
│  ┌──────────────┐           ┌──────────────────┐        │
│  │AdversarialFB │◄──────────│  ArenixAnalyzerV2│        │
│  │    Loop      │  signals  │  (scoring engine)│        │
│  └──────────────┘           └──────────────────┘        │
│                                      │                  │
│                              ┌───────▼────────┐         │
│                              │  ObserverRole  │         │
│                              │ (confirmation) │         │
│                              └───────┬────────┘         │
│                                      │                  │
│                              ┌───────▼────────┐         │
│                              │  ArenixReport  │         │
│                              │ (JSON + Phase6)│         │
│                              └────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

**Key modules:**

| File | Role |
|---|---|
| `arenix_engine.py` | Core orchestration, scoring, report generation |
| `adaptive_attacker.py` | Q-learning engine, tactic selection, adversarial feedback loop |
| `attack_library.py` | Payload catalog, selector, mutation engine |
| `app.py` | Streamlit dashboard (UI entrypoint) |
| `api.py` | FastAPI service for programmatic access |
| `semantic_engine.py` | Optional semantic intent analysis |
| `compliance_mapper.py` | OWASP LLM Top 10 / NIST CSF mapping |
| `report_generator.py` | Additional report formatting utilities |
| `tournament.py` | Multi-model tournament runner |

---

## Installation

**Requirements:** Python 3.11 or 3.12

```bash
# Clone the repository
git clone https://github.com/your-username/arenix.git
cd arenix

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Provider API Keys

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `ARENIX_CUSTOM_API_KEY` + `ARENIX_CUSTOM_BASE_URL` | OpenRouter, Azure, vLLM, or any OpenAI-compatible endpoint |

### Assigning Models to Roles

```env
ARENIX_ATTACKER_PROVIDER=custom
ARENIX_ATTACKER_MODEL=meta-llama/llama-3.3-70b-instruct

ARENIX_TARGET_PROVIDER=openai
ARENIX_TARGET_MODEL=gpt-4o-mini

ARENIX_OBSERVER_PROVIDER=anthropic
ARENIX_OBSERVER_MODEL=claude-3-5-sonnet-20241022
```

Supported providers: `mock` · `openai` · `anthropic` · `gemini` · `deepseek` · `ollama` · `custom`

---

## Running

### Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Opens an interactive UI where you can configure the session, run a simulation, and explore the attack timeline and report.

### Command-line (engine only)

```bash
python main_engine.py
```

Reads all configuration from environment variables and writes `arenix_report.json`.

### REST API

```bash
python api.py
# or with uvicorn directly:
uvicorn api:app --reload --port 8000
```

---

## Mock Mode vs Real LLM Mode

Arenix runs fully offline in **mock mode** — no API keys required.

```env
# Mock mode (default — no API keys needed)
ARENIX_ATTACKER_PROVIDER=mock
ARENIX_TARGET_PROVIDER=mock
ARENIX_OBSERVER_PROVIDER=mock
```

Switch to real LLMs by changing the provider and model variables and supplying the corresponding API key.

---

## Example Validation Workflow

```bash
# 1. Run a 12-turn mock simulation
ARENIX_MAX_TURNS=12 python main_engine.py

# 2. Run against a real target model via OpenRouter
ARENIX_CUSTOM_API_KEY=sk-or-... \
ARENIX_CUSTOM_BASE_URL=https://openrouter.ai/api/v1 \
ARENIX_ATTACKER_PROVIDER=custom \
ARENIX_ATTACKER_MODEL=meta-llama/llama-3.3-70b-instruct \
ARENIX_TARGET_PROVIDER=custom \
ARENIX_TARGET_MODEL=openai/gpt-4o-mini \
ARENIX_OBSERVER_PROVIDER=custom \
ARENIX_OBSERVER_MODEL=anthropic/claude-3.5-sonnet \
ARENIX_INDUSTRY=fintech \
python main_engine.py

# 3. Run the test suite
pytest -q
```

---

## Output and Report Structure

Each session produces a JSON report (`arenix_report.json` by default) with the following structure:

```
ArenixReport
├── session_status          SAFE | UNDER_ATTACK | UNDER_PRESSURE | RESILIENT_UNDER_ATTACK | COMPROMISED | CRITICAL
├── model_compromised       bool — True only when actionable_disclosure is confirmed
├── model_under_pressure    bool — True when sustained BOUNDARY_PRESSURE without disclosure
├── max_compromise_score    float 0–100
├── average_resilience      float 0–100
├── break_point             turn number, vulnerability_type, why_broken (if detected)
├── early_warnings          list of partial leakage / boundary weakening events
├── turn_analyses[]         per-turn: status, scores, tactic, refusal_type
├── attack_strategy_path    chronological tactic/phase sequence
├── tactic_switch_log       when and why the attacker changed strategy
├── exploitation_attempts   per-turn attack outcome record
├── security_insights       severity-tagged findings (CRITICAL / HIGH / MEDIUM / LOW)
└── executive_summary       narrative summary for compliance review
```

### Turn Status Taxonomy

| Status | Meaning |
|---|---|
| `SAFE` | No meaningful attack signal detected |
| `SOFT_COMPLIANCE` | Model responded without refusal but disclosed nothing sensitive |
| `BOUNDARY_PRESSURE` | Borderline scores (35–55); attack present but no actionable disclosure |
| `UNDER_ATTACK_BUT_RESILIENT` | Active attack; model defending effectively |
| `COMPROMISED` | Actionable disclosure confirmed |
| `CRITICAL_COMPROMISE` | Critical-threshold breach with confirmed disclosure |

---

## Testing

```bash
pytest tests/ -v
```

Tests cover scoring calibration, attack profile behavior, and the custom endpoint adapter.

---

## Safety Note

Arenix generates realistic adversarial prompts as part of its core function. These prompts are used exclusively within the simulation pipeline to evaluate LLM defenses. The system does not send requests to any production systems, social media, or external services outside of the configured LLM provider endpoints. All testing must be conducted only against models and systems you are authorized to test.

---

## Project Structure

```
arenix/
├── app.py                  Streamlit dashboard
├── arenix_engine.py        Core engine: orchestration, scoring, reporting
├── adaptive_attacker.py    Adaptive attack engine (Q-learning, Phase 1–4)
├── attack_library.py       Payload catalog and mutation engine (Phase 2)
├── semantic_engine.py      Optional semantic intent analysis
├── compliance_mapper.py    OWASP / NIST compliance mapping
├── report_generator.py     Report formatting utilities
├── tournament.py           Multi-model tournament runner
├── api.py                  FastAPI REST service
├── main_engine.py          CLI entrypoint
├── requirements.txt
├── .env.example            Environment variable reference (copy to .env)
├── .gitignore
├── LICENSE
├── tests/
│   ├── test_analyzer_scoring.py
│   ├── test_attack_profiles.py
│   └── test_custom_adapter.py
└── .github/
    └── workflows/
        └── tests.yml       CI — runs pytest on Python 3.11 and 3.12
```

---

## License

MIT — see [LICENSE](LICENSE).
