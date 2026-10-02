# CMN-C2-659 — Voice-AI Compliance & Deployment Q&A Agent

> **Category**: Cat 2 (a domain pipeline for one specific job-to-be-done)
> **Industry**: CMN (industry-neutral)

## Overview

Answers questions about deploying a voice AI agent in Japan and checks a declared deployment
profile against the rules that apply to it.

Ask it a question — in Japanese or English — about voice-recording consent and retention, about
accessibility obligations for a voice interface, about multilingual coverage for a workforce
deployment, or about choosing an open-source voice model. It identifies which Japanese laws bear
on the question, retrieves supporting passages, and returns a structured guidance document.

Send it your deployment parameters as well — retention period, deletion service level,
speech-rate range, recognition accuracy, response latency, supported languages — and it evaluates
each one against the corresponding threshold, reports the shortfall where there is one, and grades
the answer overall as `info`, `attention` or `action_required`. You can also supply your own
knowledge-base entries; the agent scores them against your question and answers from them
alongside its built-in corpus.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails during
graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling the agent

`POST /invoke` takes the question in `input` and, optionally, structured parameters in
`input_context`:

```json
{
  "input": "音声AIの録音保持期間とアクセシビリティ要件を教えてください",
  "session_id": "optional",
  "input_context": {
    "channel": "web",
    "locale": "ja",
    "deployment_profile": {
      "recording_retention_days": 180,
      "deletion_sla_days": 45,
      "intent_accuracy": 0.81,
      "response_latency_ms": 450,
      "speech_rate_min": 0.75,
      "speech_rate_max": 1.5,
      "supported_languages": ["japanese", "mandarin"]
    },
    "kb_documents": [
      {"doc_id": "internal_policy_01", "content": "Our retention policy states ..."}
    ]
  }
}
```

Everything in `input_context` is optional. With none of it the agent answers from its built-in
corpus; with a `deployment_profile` it also evaluates each declared value and grades the answer
`info`, `attention` or `action_required`; with `kb_documents` it answers from your entries as well.

The contract is bounded field by field, and a rejection names the field and never repeats the
value you sent:

| Field | Rule |
|---|---|
| `channel`, `locale` | lowercase identifier, `a-z 0-9 _`, 1–32 characters |
| `deployment_profile.*` numbers | finite (never `NaN` or `Infinity`) and inside a declared range |
| `deployment_profile.supported_languages` | up to 20 lowercase identifiers |
| `kb_documents` | up to 20 entries; `doc_id` a lowercase identifier, `content` up to 4000 characters |
| whole payload | up to 256 KB serialized |

Set `INVOKE_AUTH_TOKEN` in the server environment to require `Authorization: Bearer <token>` on
`/invoke`. The agent's first node requires a verified caller, so a deployment that leaves the
variable unset and puts no authenticating middleware in front of the app will refuse every request.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
