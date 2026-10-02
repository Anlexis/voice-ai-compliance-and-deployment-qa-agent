# Template Design Specification — CMN-C2-659
# Voice-AI Compliance & Deployment Q&A Agent

## Position in the Framework

| Aspect | Value |
|---|---|
| Agent class | `VoiceAIComplianceDeploymentQAAgent` (matches `config/agent.yaml` `class:` and the `src/api/server.py` import) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Category | Cat 2 — a multi-step domain workflow for one job-to-be-done |
| Three-layer separation | State: flat TypedDict (`src/schemas/state.py`, no Pydantic — checkpoints are msgpack-serialised). Node: framework inheritance, `execute(self, state, config=None) -> dict`. Graph: composition — `register_nodes()` assigns nodes, `add_edges()` wires the inner topology. |

## Architecture

### Two-layer nested pattern

```
Outer backbone (AgentBaseGraph — fixed, NOT overridden):
  START → initialize → pre_process → main → {route} → post_process → finalize → END
                                         ↓ (RETRY, max_retry)
                                      pre_process

  main slot = VoiceAIComplianceWorkflowGraphNode (GraphNode subclass)
           ↓ get_subgraph() → DomainWorkflowGraph (inner BaseGraph)

Inner domain workflow (DomainWorkflowGraph — fully custom topology):
  START → question_parse → legal_context_identify → hybrid_retrieve
        → appi_voice_annotate → accessibility_check_format → model_recommend
        → response_compose → security_gate_output → END
```

### Node configuration

| Node | Class | Slot | Trust | Responsibility |
|------|-------|------|-------|----------------|
| initialize | `InitializeNode` (framework) | initialize | — | schema version, session id, caller trust level |
| pre_process | `PreProcessNode` | outer pre_process | `VERIFIED_EXTERNAL` | the whole caller contract: question screening and `input_context` validation |
| main | `VoiceAIComplianceWorkflowGraphNode` | outer main | `ANONYMOUS` | delegates to `DomainWorkflowGraph`; carries the validated payload inward |
| post_process | `PostProcessNode` | outer post_process | `ANONYMOUS` | `result` → `formatted_output` |
| finalize | `FinalizeNode` (framework) | finalize | — | response metadata, total time |
| question_parse | `QuestionParseNode` | inner | `ANONYMOUS` | normalise and classify the question |
| legal_context_identify | `LegalContextIdentifyNode` | inner | `ANONYMOUS` | identify applicable Japanese law |
| hybrid_retrieve | `HybridRetrieveNode` | inner | `ANONYMOUS` | score caller entries against the question, merge with the baseline corpus |
| appi_voice_annotate | `APPIVoiceAnnotateNode` | inner | `ANONYMOUS` | recording-governance obligations; evaluate declared retention and deletion values |
| accessibility_check_format | `AccessibilityCheckFormatNode` | inner | `ANONYMOUS` | accessibility obligations; evaluate the declared speech-rate range |
| model_recommend | `ModelRecommendNode` | inner | `ANONYMOUS` | model guidance; evaluate declared accuracy, latency and language coverage |
| response_compose | `ResponseComposeNode` | inner | `ANONYMOUS` | compose the document; derive the overall severity |
| security_gate_output | `SecurityGateOutputNode` | inner (last) | `ANONYMOUS` | the output boundary |

### Caller-data contract

`/invoke` accepts the question in `input` and structured parameters in `input_context`. Everything
is optional; with none of it the agent answers from its built-in corpus alone.

| Field | Type | Bound |
|---|---|---|
| `channel`, `locale` | string | inert identifier `[a-z0-9_]{1,32}` |
| `deployment_profile.recording_retention_days` | number | finite, 0–3650 |
| `deployment_profile.deletion_sla_days` | number | finite, 0–3650 |
| `deployment_profile.intent_accuracy` | number | finite, 0–1 |
| `deployment_profile.response_latency_ms` | number | finite, 0–60000 |
| `deployment_profile.speech_rate_min` / `speech_rate_max` | number | finite, 0.1–10.0, min ≤ max |
| `deployment_profile.supported_languages` | array | ≤ 20 inert identifiers |
| `kb_documents` | array | ≤ 20 entries; `doc_id` an inert identifier, `content` ≤ 4000 characters |
| whole payload | — | ≤ 256 KB serialized (enforced at the entry point) |

Unknown keys are ignored rather than refused — the hosting platform puts its own material on this
channel — but every string leaf, KEYS included, is screened before any field is interpreted.

**Crossing the graph boundary.** The framework's `GraphNode` invokes a subgraph as
`subgraph.invoke(user_input, session_id=…, ctx=…)` and does not forward `input_context`. Without a
bridge, every inner-node read of the caller's data returns nothing and the agent silently answers
from its baseline while unit tests that hand a node a populated dict keep passing.
`src/graph/context_bridge.py` closes it with a ContextVar: the outer node's `extract_input()` stashes
the VALIDATED payload, and `DomainWorkflowGraph._extra_initial_state()` seeds it into the inner
initial state. Raw caller data never crosses — it is bounded once, upstream of the bridge.

**Runtime settings.** `config/config.yaml` holds `max_retry` and `timeout_s`. The platform registry
loads that file and passes it as `Graph(config=…)`; `src/api/server.py` calls `runtime_config()` and
does the same, and the outer graph node forwards it to the inner graph. Without both, a value
declared in `config.yaml` is inert in one of the two deployments and nothing reports it.

### State (`src/schemas/state.py`)

| Field | Type | Purpose | Set by |
|-------|------|---------|--------|
| `validated_input` | `Optional[str]` | screened question | PreProcessNode |
| `enriched_context` | `Optional[dict]` | channel / locale metadata | PreProcessNode |
| `caller_profile` | `Optional[dict]` | validated deployment parameters | PreProcessNode |
| `caller_documents` | `Optional[list]` | validated caller knowledge-base entries | PreProcessNode |
| `parsed_question` | `Optional[str]` | normalised question | QuestionParseNode |
| `question_type` | `Optional[str]` | `legal_compliance` / `appi` / `accessibility` / `technical_config` / `general` | QuestionParseNode |
| `legal_contexts` | `Optional[list]` | applicable laws | LegalContextIdentifyNode |
| `retrieval_results` | `Optional[list]` | scored excerpts | HybridRetrieveNode |
| `appi_annotations` | `Optional[str]` | recording-governance section | APPIVoiceAnnotateNode |
| `accessibility_notes` | `Optional[str]` | accessibility section | AccessibilityCheckFormatNode |
| `model_recommendations` | `Optional[str]` | model-guidance section | ModelRecommendNode |
| `compliance_findings` | `Optional[list]` | `{area, code, severity, detail}` per evaluated value | the evaluation nodes |
| `overall_severity` | `Optional[str]` | strongest finding severity | ResponseComposeNode |
| `draft_response` | `Optional[str]` | composed document | ResponseComposeNode |
| `secured_response` | `Optional[str]` | released document | SecurityGateOutputNode |
| `result` | `Optional[str]` | outer state — mapped from the inner output | `merge_output()` |
| `formatted_output` | `Optional[str]` | final output | PostProcessNode |

Constraints: flat TypedDict only; JSON-serialisable primitives; no credentials in State; no Pydantic
models, dataclasses or arbitrary objects.

## Security model

### Trust

`PreProcessNode` declares `VERIFIED_EXTERNAL`; every other node declares `ANONYMOUS`, so the caller
check happens once, at the outer boundary, and inner nodes admit whatever the outer gate admitted.
The entry point promotes a caller presenting a valid `INVOKE_AUTH_TOKEN` bearer credential to
`VERIFIED_EXTERNAL`; middleware-established trust is never demoted.

### Input contract (PreProcessNode)

1. Type guard, blank guard, size cap on the question.
2. Credential screen using the framework detector, so this template's refusal set and the
   platform's block set are one set by construction.
3. Instruction-override screen in three classes: chat-template control tokens (`<|…|>`, `[INST]`,
   `<<SYS>>`, `<system>`), anchored override phrases, and — for the question only — a forged
   conversation turn at line start. Each string is screened as received AND after markup removal,
   so a directive split by inert tags is caught once it reassembles.
4. Personal-data screen (email, structured phone number, payment card). Every phone alternative
   requires country-code, trunk-prefix or separator structure: a bare digit run is ordinary
   technical prose in this domain, and refusing it would be the fail-closed defect that actually
   blocks work.
5. `input_context`: depth-first screen of every string leaf, keys included, before any field is
   interpreted; then per-field bounds — inert identifiers for labels, `finite_in_range` for every
   number, entry caps for every list.

Every refusal names the field and never echoes the value.

**Why the template owns this rather than the platform.** The framework's input gate covers only
`user_input` / `validated_input`, rejects only high-confidence findings, and is absent on older
hosts. A template whose only defence is that gate returns success on an injection payload wherever
it is absent — it fails open. The refusal above is the template's own guarantee, and the tests prove
it by calling `execute()` directly.

### The credential hazard on the context channel

The framework's output gate scans every value of every node result, and the backbone's initialize
node copies `input_context` verbatim into its own result. A credential-shaped string anywhere in
`input_context` therefore fails the FIRST node of the graph, before any template code runs, and the
caller gets an error naming nothing. `src/api/server.py` screens the assembled context with the same
framework detector before `invoke()` and refuses with a 400 that names the offending field. The
request could not have succeeded either way; the screen turns an opaque failure into an actionable
one.

### Output boundary (SecurityGateOutputNode)

The stated invariant: **the released document carries no credential-shaped and no
personal-data-shaped string, and every caller-supplied label it reproduces is an inert identifier.**

Two independent layers, each with its own audit event:

1. **Pattern scan** over the composed document AND, depth-first, over the nested structures it was
   composed from — caller text arrives inside lists of mappings and only part of it is quoted, so a
   top-level-only scan reports nothing for a credential one level down or just past the excerpt cut.
2. **Label check** re-validating every caller identifier against the inert alphabet, independently
   of the upstream validation: an invariant this layer states is one this layer can enforce without
   trusting the node in front of it.

**A monetary rounding grid is not applicable here.** This agent renders no monetary aggregates, so
there is nothing for such a grid to round. Its absence also removes the failure mode that comes with
one: a numeric snap rewrites standalone digit runs, which destroys the shape of an identifier or an
account number immediately before a pattern scan would have caught it. With no snap in the pipeline,
the scan sees the text exactly as composed, and structural numbers — day counts, millisecond
budgets, years, version strings, decimal ratios — reach the caller byte-identical. The rendering
rules the document does state (ratios as percentages to one decimal place, day and millisecond
counts as whole numbers) are printed in the document itself.

**Withholding is containment, not a flag.** A node returns a partial state update, so a field the
update omits keeps the value it had — and callers resolve the document through
`secured_response or draft_response`, which consults no status. On a violation the boundary returns
an error status AND replaces both carrying fields with a fixed, truthy notice. Two further layers
are independent of it: `DomainWorkflowGraph.get_output()` returns a document only on a successful
run, and `VoiceAIComplianceDeploymentQAAgent.get_output()` does the same for the outer envelope.
Any one of the three contains the leak on its own.

### Audit

`emit_trace_event` is called on every path of every node. Event types:
`pre_process_validated`, `pre_process_rejected_type` / `_empty` / `_size` / `_credential` /
`_injection` / `_personal_data` / `_context`, `input_context_credential_refused`, `question_parsed`,
`legal_contexts_identified`, `hybrid_retrieve_completed`, `appi_annotation_evaluated`,
`accessibility_check_evaluated`, `model_recommendations_generated`, `response_composed`,
`output_gate_passed` / `output_gate_blocked` / `output_gate_error`, `post_process_completed`.

## Framework utilisation

- `InvocationContext` — passed from the outer backbone into the inner graph by the framework's
  `GraphNode`.
- Input and output gate extensions are module-level helpers called from `execute()`; the framework's
  own gates of those names are final and are never overridden.
- `emit_trace_event()` — imported as a free function, never through `self`.

Composition: `GraphNode` subgraph, `error_strategy = "propagate"`, human-in-the-loop not enabled.

## Import isolation

- No platform SDK imports; `framework/` and `shared/` only.
- `DomainWorkflowGraph` is imported lazily inside `get_subgraph()` to avoid a load-time cycle.

## Design decisions

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Outer base class | `AgentBaseGraph` | `AutonomousBaseGraph` | `AgentBaseGraph` | fixed 8-step pipeline; no self-directed loop |
| Inner graph | `BaseGraph` | `AgentBaseGraph` | `BaseGraph` | custom 8-node topology with no backbone slots |
| Inner node trust | `ANONYMOUS` | `INTERNAL` | `ANONYMOUS` | the outer gate admits the caller once; `INTERNAL` would deny the very callers it admitted |
| Caller data across the boundary | ContextVar bridge | forward `input_context` | bridge | the framework's `GraphNode` does not forward it, and the bridge carries validated data only |
| Withholding | clear the fields | return an error status | clear the fields | a partial update leaves the refused document in place for the fallback to find |
| Caller retrieval slots | reserved | pure ranking | reserved | fixed baseline scores would otherwise fill every slot and answer a caller entirely from our corpus |
