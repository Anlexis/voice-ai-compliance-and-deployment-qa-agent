"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — SecurityGateOutputNode
# Inner workflow node 8 (final): the output boundary.
#
# The stated output invariant of this agent is:
#
#   the released guidance document carries no credential-shaped and no
#   personal-data-shaped string, and every caller-supplied label it reproduces
#   is an inert identifier.
#
# There is no monetary rounding grid here: this agent renders no monetary
# aggregates at all, so the grid that other templates enforce has nothing to
# apply to. Its absence also removes the failure mode that comes with it — a
# numeric snap rewrites any standalone digit run, which destroys the shape of
# an identifier or an account number just before a pattern scan would have
# caught it. With no snap in the pipeline, the pattern scan below sees the text
# exactly as composed.
#
# Two independent layers, each with its own audit event:
#   1. pattern scan  — credential and personal-data shapes, over the composed
#                      document AND every nested value it was composed from;
#   2. label check   — caller-supplied identifiers re-validated against the
#                      inert alphabet, independently of the upstream validation.
#
# Input state keys:
#   draft_response:   str  — composed document from ResponseComposeNode
#   caller_documents: list — validated caller entries (labels re-checked here)
#   caller_profile:   dict — validated caller profile (labels re-checked here)
#   retrieval_results: list — nested structures the document was composed from
#
# Output state keys (partial dict):
#   secured_response: str  — the released document
#   draft_response:   str  — replaced by the withheld notice on violation
#   status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:        list[str] (ERROR only)
#
# Security notes:
#   Audit: emit_trace_event on every path.
#   Trust: ANONYMOUS — inner domain node.

import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import IDENTIFIER_PATTERN

# ── Withholding ───────────────────────────────────────────────────────────────

# What replaces the document in state when the boundary refuses to release it.
#
# It has to be non-empty. Callers resolve the document through expressions of
# the shape `secured_response or draft_response`, which consult no status at
# all, so an empty string or None is not a cleared field — it is falsy, the
# fallback fires, and the refused document ships inside an envelope that says
# the run failed. A truthy placeholder occupies the field and no fallback can
# reach past it.
#
# It is fixed and inert. Naming the violated rule is the job of the audit event
# and the error log; anything derived from the document would put the withheld
# content back on the very surface the withholding exists to protect.
WITHHELD_NOTICE = "[withheld] The guidance document was withheld at the output boundary."

# Every field that can carry the document, replaced as one unit.
#
# An error status alone is not containment: a node returns a PARTIAL state
# update, so a field the update omits keeps the value it already had.
# draft_response is what the boundary just refused and secured_response is what
# a caller reads, so both are replaced together.
_CLEARED_ON_WITHHOLD: Dict[str, Any] = {
    "draft_response": WITHHELD_NOTICE,
    "secured_response": WITHHELD_NOTICE,
}

# ── Layer 1: patterns that must never appear in a released document ───────────

# Two shapes, kept apart because they fail differently:
#   (a) a NAMED secret — `api_key: …`, `password = …` — which the platform
#       detector does not recognise, so this layer is the only one that sees it;
#   (b) a literal token shape — key prefixes, JWTs, bearer values, connection
#       strings — which the entry point also refuses, deliberately: the entry
#       point cannot see text this pipeline composes for itself.
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)("
    r"(?:password|passwd|api[_\-]?key|secret[_\-]?key|access[_\-]?key|auth[_\-]?token)"
    r"\s*[=:]\s*\S+"
    r"|bearer\s+[A-Za-z0-9\-._~+/]{16,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|sk_(?:live|test)_[A-Za-z0-9]{16,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|(?:postgresql|mysql|mongodb|redis)://\S{8,}"
    r")"
)

# The same phone forms the input contract screens for. The boundary is a
# backstop, so it must not recognise less than the layer in front of it — but it
# must not recognise MORE either: an unanchored digit run here would withhold
# documents over corpus sizes and parameter counts.
_PERSONAL_DATA_PATTERN = re.compile(
    r"(?:"
    r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}"  # email
    r"|\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"  # North-American, country code
    r"|\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}"  # North-American, separated
    r"|\+81[-.\s]?\d{1,4}[-.\s]?\d{2,4}[-.\s]?\d{4}"  # Japan international
    r"|0\d{1,4}[-.\s]\d{2,4}[-.\s]\d{3,4}"  # Japan domestic, separated
    r"|0[789]0\d{8}"  # Japan mobile, unseparated
    r"|\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"  # payment card
    r")"
)

# ── Layer 2: the inert-label invariant ────────────────────────────────────────

_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)


def _security_gate_output(text: str) -> Optional[str]:
    """Scan released text for credential and personal-data shapes.

    Returns None when the text is safe to release, or the violation CLASS when
    it is not. The class name is what travels into the audit event and the error
    log; the matched text never does.

    This is a module-level function, deliberately not a method: the framework's
    own output gate of the same name is final on FunctionNode, and a subclass
    that tried to override it would be rejected at class definition time. This
    helper runs inside execute() as the domain layer on top of that gate.
    """
    if _CREDENTIAL_PATTERN.search(text):
        return "credential-shaped value in the released document"
    if _PERSONAL_DATA_PATTERN.search(text):
        return "personal data in the released document"
    return None


def scan_value(value: object) -> Optional[str]:
    """Depth-first scan of a nested structure for the same patterns.

    Caller text reaches this node inside lists of mappings, and only part of it
    is quoted into the document. A scan that looked at top-level strings alone
    would report nothing for a credential that sits one level down or just past
    the excerpt cut, and the answer would ship with the source it was drawn from
    still unscreened.
    """
    if isinstance(value, str):
        return _security_gate_output(value)
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found = _security_gate_output(key)
                if found:
                    return found
            found = scan_value(item)
            if found:
                return found
        return None
    if isinstance(value, list):
        for item in value:
            found = scan_value(item)
            if found:
                return found
    return None


def check_labels(caller_documents: Any, caller_profile: Any) -> Optional[str]:
    """Re-validate every caller label that the document reproduces.

    The caller contract already enforces this alphabet at the input boundary.
    Checking it again here is not redundancy for its own sake: this is the layer
    that owns the released document, and an invariant the output boundary states
    should be one the output boundary can enforce without trusting an upstream
    node to have done it.
    """
    if isinstance(caller_documents, list):
        for document in caller_documents:
            if not isinstance(document, dict):
                return "caller knowledge-base entry is not an object"
            doc_id = document.get("doc_id")
            if not isinstance(doc_id, str) or not _IDENTIFIER_RE.match(doc_id):
                return "caller knowledge-base label is not an inert identifier"
    if isinstance(caller_profile, dict):
        languages = caller_profile.get("supported_languages")
        if languages is not None:
            if not isinstance(languages, list):
                return "caller language list is not an array"
            for entry in languages:
                if not isinstance(entry, str) or not _IDENTIFIER_RE.match(entry):
                    return "caller language label is not an inert identifier"
    return None


class SecurityGateOutputNode(FunctionNode):
    """Output boundary — the final inner workflow node for CMN-C2-659.

    Enforces the agent's stated output invariant across both layers, releases
    the document when it holds, and withholds it — status AND every carrying
    field — when it does not.

    Output (partial dict — only changed keys):
        secured_response, draft_response, status, error_log.
    """

    # Inner domain node — ANONYMOUS; the outer gate owns the caller check.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error_log: List[str] = list(state.get("error_log") or [])
        draft_response: str = state.get("draft_response", "")

        if not draft_response:
            emit_trace_event(
                "output_gate_error",
                {"reason": "draft_response is absent"},
                state,
            )
            return {
                **_CLEARED_ON_WITHHOLD,
                "status": AgentStatus.ERROR.value,
                "error_log": error_log
                + ["SecurityGateOutputNode: draft_response is absent; ResponseComposeNode must run first"],
            }

        # ── Layer 1 — pattern scan over the document and its sources ──────────
        violation = _security_gate_output(draft_response)
        if violation is None:
            violation = scan_value(state.get("retrieval_results"))
        if violation:
            return self._withhold(state, error_log, violation, "pattern_scan")

        # ── Layer 2 — the inert-label invariant ───────────────────────────────
        label_violation = check_labels(state.get("caller_documents"), state.get("caller_profile"))
        if label_violation:
            return self._withhold(state, error_log, label_violation, "label_check")

        emit_trace_event(
            "output_gate_passed",
            {"response_length": len(draft_response)},
            state,
        )

        return {
            "secured_response": draft_response,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }

    def _withhold(self, state: Dict[str, Any], error_log: List[str], violation: str, layer: str) -> Dict[str, Any]:
        """Withhold the document: error status AND every carrying field replaced.

        The message names the violation class and the layer that raised it and
        quotes nothing from the document. That is not only a logging
        convention: the framework scans every value of this returned mapping for
        credential patterns, and a message quoting the matched text would trip
        it. The framework then converts the raised error into its own bare error
        update — which clears nothing — so echoing the finding would discard the
        clearing performed here and release the document it was meant to
        withhold.
        """
        emit_trace_event(
            "output_gate_blocked",
            {"violation": violation, "layer": layer, "response_length": len(state.get("draft_response", ""))},
            state,
        )
        return {
            **_CLEARED_ON_WITHHOLD,
            "status": AgentStatus.ERROR.value,
            "error_log": error_log + [f"SecurityGateOutputNode: output withheld — {violation}"],
        }
