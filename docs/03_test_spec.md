# Test Specification — CMN-C2-659
# Voice-AI Compliance & Deployment Q&A Agent

## Overview

Every test below ships in this repository and runs against the real
`agenticstar-agentcore` wheel that CI installs (`1.0.2`). Assertions are behavioural: a refusal is
an error status with the offending value absent from what comes back. Nothing asserts a framework's
error wording, because that wording is not this template's guarantee and it changes between wheel
versions.

Layout:

| File | Covers |
|---|---|
| `tests/unit/test_pre_process_node.py` | the caller contract |
| `tests/unit/test_domain_nodes.py` | the eight inner nodes and the boundary helpers |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | the non-bypassable framework gates |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | backbone order and trust denial |
| `tests/proof_of_boundary/test_pb_invoke_e2e.py` | the whole agent through the real HTTP entry point |
| `tests/proof_of_boundary/test_import_isolation.py` | no platform-SDK imports |
| `tests/proof_of_boundary/test_state_safety.py` | State carries no credentials and no unsafe types |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | skip stub — this template wires no cross-boundary interrupt |

## Framework compliance

| TC-ID | Test | Expected |
|-------|------|----------|
| TC-01 | State is a flat TypedDict (no Pydantic, no dataclass, no credential-shaped field names) | `test_state_safety.py` reports no violations |
| TC-02 | The default input gate cannot be overridden on a `FunctionNode` subclass | `TypeError` at class definition |
| TC-03 | The default output gate cannot be overridden on a `FunctionNode` subclass | `TypeError` at class definition |
| TC-04 | Every `FunctionNode` subclass declares `required_trust_level` | asserted per class |
| TC-05 | Every node emits at least one domain audit event on a reachable path | asserted by the repository's audit-trace check |
| TC-06 | No platform-SDK imports under `src/` | `test_import_isolation.py` |
| TC-07 | No credentials in State or source | repository credential check |

## Unit tests — the caller contract

`tests/unit/test_pre_process_node.py`

| Group | Cases |
|---|---|
| Trust declaration | `required_trust_level == VERIFIED_EXTERNAL`; `execute(self, state, …)` signature |
| Question contract | valid question accepted and trimmed; non-string (dict / None / list / int / bool) refused; blank and whitespace-only refused; over 8000 characters refused; credential-shaped question refused without echoing the value |
| Directive screen — hostile | `<\|im_start\|>`, `<\|endoftext\|>`, `[INST]`, `<<SYS>>`, `<system>`, "ignore all previous instructions", "disregard prior instructions", "forget all your instructions", "you are now a…", "pretend to be…", 前の指示を無視, a forged `system:` turn |
| Directive screen — spliced | `ig<b>nore</b> all previous <i>instructions</i>` refused after markup removal |
| Directive screen — benign | five real domain sentences that contain the same words ("録音の保持期間を無視してよいケース", "act as the data controller", "ignore the previous estimate of latency", …) all accepted |
| Directive screen — asymmetry | a document may label a section `System:`; a question may not |
| Personal data — hostile | email, corporate email, `090-1234-5678`, `09012345678`, payment card |
| Personal data — benign | 11-digit parameter counts and corpus sizes are not phone numbers |
| Profile — accepted | every declared value carried through with the parsed type |
| Profile — non-finite | 6 numeric fields × 6 non-finite forms (`"NaN"`, `"Infinity"`, `"-Infinity"`, `float("nan")`, `float("inf")`, `float("-inf")`) = 36 cases, each refused with the field named |
| Profile — out of range | below minimum, above maximum, per field |
| Profile — wrong type | booleans, strings and objects refused |
| Profile — cross-field | `speech_rate_min > speech_rate_max` refused |
| Profile — labels | non-inert language labels refused; list capped at 20; `channel` / `locale` must be inert |
| Documents | valid entries carried through; non-inert `doc_id` refused; list capped at 20; content capped at 4000; personal data in content refused; empty or non-string content refused |
| Context screen | directive in a nested value; directive in a field NAME; `\u`-escaped payload caught post-parse; unknown keys ignored but their content still screened |
| Finite parser | non-finite forms, booleans, and in-range values |

## Unit tests — the domain nodes

`tests/unit/test_domain_nodes.py`

| Group | Cases |
|---|---|
| Node contracts | all nine nodes declare a trust level and take `state` first |
| Question parse | five classification cases; whitespace collapse; absent input is an error |
| Legal context | recording question matches the recording rules; no match is still a success; absent question is an error |
| Retrieval | no caller corpus degrades to the baseline; a matching caller entry is not crowded out; an unrelated entry is left out; score rises with term coverage; score is zero without terms |
| Recording governance | no profile yields no findings; a 180-day retention reports "90 days beyond"; a 100-day retention is `attention` not `action_required`; a compliant period is `info`; a 45-day deletion level reports "15 beyond"; prior findings are preserved |
| Accessibility | a 0.75–1.5 range is reported at both ends; a 0.5–2.0 range is `info` |
| Model | 81% accuracy reports "4.0 percentage points"; 450 ms reports "150ms above" at `attention`; 900 ms escalates to `action_required`; missing languages are named; full coverage is `info` |
| Compose | overall severity is the strongest finding; no findings yields `info` and says why; the document states its own rendering rules; absent question is an error |
| Output boundary | clean text released; 6 credential shapes withheld; 3 personal-data shapes withheld; a withheld document is REPLACED not merely flagged; the replacement is truthy; the message quotes nothing; a credential nested in the sources is caught; the clean-sources control passes; a credential in a nested KEY is caught; absent document clears the fields; the helper is a module function not an override; labels are re-validated; a label violation withholds; structural numbers (day counts, millisecond budgets, years, `0.5x-2.0x`, `8.512345%`) survive byte-identical |
| Post process | `result` forwarded; absent `result` tolerated |

## Proof-of-boundary tests

### PB-6 — backbone order and trust

`tests/proof_of_boundary/test_pb_invoke_order.py`

| PB-ID | Description | Expected |
|-------|-------------|----------|
| PB-6-01 | A verified caller gets a successful invoke | `status == SUCCESS`, `output` present |
| PB-6-02 | `PreProcessNode` runs before the main slot node | verified through `node_history` |
| PB-6-03 | An anonymous caller is denied | non-success, or no output |

The payload used here is the same string as `deploy/invoke_payload.json`.

### End-to-end through the HTTP entry point

`tests/proof_of_boundary/test_pb_invoke_e2e.py` — the real ASGI app, the real compiled graph,
bearer authentication enabled.

| Group | Cases |
|---|---|
| Entry point | health; unauthenticated refused (401); wrong token refused without saying why; an authenticated caller reaches the pipeline; oversized context refused (413) |
| Baseline answer | the answer is a real document with all six sections; the backbone ran in order; with no caller data the answer says so and grades `info` |
| Caller data reaches the graph | a declared profile changes the verdict to `action_required` and reproduces the arithmetic each declared number implies; a compliant profile reaches `info`; the middle severity `attention` is reachable; a caller knowledge-base entry is quoted in the answer; an empty context degrades to the baseline |
| Contract rejections | non-finite strings; a bare `NaN` literal on the wire; out-of-range; free-text label; control token in the question; control token in the context; an ordinary question containing the same words still succeeds |
| Credential screen | a credential in the context is refused 400 with the field named and the value absent; a credential in the question is refused; ordinary domain text on the same field still passes |
| Output containment | the run fails; the envelope carries no document; the envelope leaks no secret, no traceback, no source path; the clean control of the same shape still succeeds |
| Released document scan | the released document passes the boundary scan; every caller label in it matches the inert alphabet |

### PB-4 — import isolation

`tests/proof_of_boundary/test_import_isolation.py` — AST scan; no platform-SDK imports under `src/`.

### PB-2 / PB-5 — state safety

`tests/proof_of_boundary/test_state_safety.py` — State declares no credential-shaped field name and
no prohibited type.

## Running

```bash
python -m pytest tests/ -v              # everything
python -m pytest tests/unit/ -v         # unit only
python -m pytest tests/proof_of_boundary/ -v
```

## Conventions

- `emit_trace_event` is patched at the NODE MODULE level (e.g.
  `src.nodes.pre_process_node.emit_trace_event`), never by stubbing the `shared.*` package — that
  would break framework imports.
- `result["output"]` is the caller-facing document, not `result["formatted_output"]`.
- Status comparisons use `AgentStatus.<X>.value`.
- `invoke()` takes `user_input`, `ctx` and `input_context`.

## Results

- Total: 229 tests (228 pass, 1 skip — the interrupt-propagation stub this template does not wire).
- Wheel: `agenticstar-agentcore==1.0.2`.
