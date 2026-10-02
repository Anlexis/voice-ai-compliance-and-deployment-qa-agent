"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — HybridRetrieveNode
# Inner workflow node 3: hybrid retrieval (keyword scoring over the applicable
# legal contexts) across two corpora:
#
#   1. the caller's own knowledge-base entries, validated upstream, and
#   2. the built-in baseline corpus.
#
# Caller entries are scored against the question's own terms, so a caller who
# brings a corpus gets answers grounded in it rather than in the baseline.
# With no caller corpus the node degrades to the baseline alone — the same
# answers the agent gave before the caller channel existed.
#
# A production deployment replaces the baseline lookup with a vector or hybrid
# search service; the caller-corpus path and the scoring are unchanged by that.
#
# Input state keys:
#   parsed_question:  str       — normalised question text
#   question_type:    str       — question category
#   legal_contexts:   list[dict]— applicable laws from LegalContextIdentifyNode
#   caller_documents: list[dict]— validated caller knowledge-base entries
#
# Output state keys (partial dict):
#   retrieval_results: list[dict] — scored excerpts, highest relevance first
#   status:            AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
#   error_log:         list[str] (ERROR only)
#
# Security notes:
#   Audit: emit_trace_event on success/error.
#   Trust: ANONYMOUS — inner domain node.

import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# ── Baseline corpus (module-level) ────────────────────────────────────────────
# Each record: {id, title, content, relevance_score}

_KB_RECORDS: Dict[str, List[Dict[str, Any]]] = {
    "APPI_2026": [
        {
            "id": "appi-v1-001",
            "title": "APPI 2026 Voice Recording Consent Requirements",
            "content": (
                "Under APPI 2026 amendments, voice AI systems that record interactions "
                "must obtain explicit prior consent from data subjects. Consent must specify "
                "the purpose of recording, retention period (maximum 90 days unless extended), "
                "and the deletion rights of the data subject. Opt-out mechanisms must be "
                "accessible at any point during the interaction."
            ),
            "relevance_score": 0.95,
        },
        {
            "id": "appi-v1-002",
            "title": "APPI 2026 Retention Period and Deletion Rights",
            "content": (
                "Voice recordings are classified as sensitive personal information under "
                "APPI 2026 if they contain biometric identifiers. Retention periods must be "
                "documented, and data subjects may request deletion. Systems must implement "
                "automated deletion workflows within 30 business days of a deletion request."
            ),
            "relevance_score": 0.90,
        },
    ],
    "BARRIER_FREE_2024": [
        {
            "id": "bf2024-001",
            "title": "Barrier-Free Act 2024 — Voice AI Accessibility Requirements",
            "content": (
                "The 2024 amendment mandates that voice AI interfaces deployed in public "
                "services and customer-facing roles provide accessibility features including: "
                "(1) speech-to-text captions, (2) Japanese sign language support via "
                "digital avatar, (3) adjustable speech rate (0.5x-2.0x), and (4) dialect "
                "recognition for regional accessibility."
            ),
            "relevance_score": 0.93,
        },
    ],
    "IKUSEI_SHURO_2028": [
        {
            "id": "ikusei-001",
            "title": "Ikusei-Shuro 2028 — Multilingual Voice AI Requirements",
            "content": (
                "Employers using voice AI for workforce communication with Ikusei-Shuro "
                "programme workers must support the worker's native language. Minimum "
                "supported languages: Mandarin, Vietnamese, Indonesian, Tagalog, and "
                "Japanese. Voice models must achieve at least 85% intent recognition "
                "accuracy for each supported language."
            ),
            "relevance_score": 0.88,
        },
    ],
    "general": [
        {
            "id": "general-deploy-001",
            "title": "Voice AI Deployment Practices for Japanese Enterprise",
            "content": (
                "Voice AI deployment in Japanese enterprise settings should consider: "
                "(1) Japanese dialect coverage (Kansai, Tohoku, Kyushu), "
                "(2) honorific speech (keigo) recognition, "
                "(3) latency requirements (under 300ms for conversational feel), "
                "(4) fallback to human operator on confidence threshold breach, "
                "(5) telephony integration via SIP trunk or WebRTC."
            ),
            "relevance_score": 0.80,
        },
        {
            "id": "general-models-001",
            "title": "Open-Source Voice AI Models for Enterprise Japan",
            "content": (
                "Open-source voice models in common enterprise use: "
                "VoxCPM (OpenBMB) — high-accuracy multi-language synthesis and recognition; "
                "MOSS-TTS (Fudan) — Japanese-optimised synthesis; "
                "VibeVoice — low-latency streaming synthesis. "
                "Evaluation criteria: word error rate on the CSJ corpus, real-time factor on "
                "standard hardware, license compatibility (Apache 2.0 / MIT preferred)."
            ),
            "relevance_score": 0.82,
        },
    ],
}

# Terms shorter than this carry no retrieval signal and match everywhere.
_MIN_TERM_LENGTH = 2

# Caller entries score in the same 0..1 band as the baseline. The ceiling sits
# just under the highest baseline score so a caller entry can lead the ranking
# on a strong term match without displacing the statutory references outright.
_CALLER_SCORE_CEILING = 0.94
_CALLER_SCORE_FLOOR = 0.40

_MAX_RESULTS = 5

# Slots reserved for matching caller entries. Without a reservation the built-in
# corpus — whose scores are fixed and high — fills every slot and a caller who
# brought a corpus is answered entirely from ours, silently. Reserving keeps the
# ranking honest for the rest of the set while guaranteeing that a caller's own
# material is represented in the answer whenever it matches at all.
_RESERVED_CALLER_SLOTS = 2

_TERM_RE = re.compile(r"[0-9A-Za-z_]+|[぀-ヿ一-鿿]+")


def _terms(text: str) -> List[str]:
    """Split a question into scoreable terms.

    Latin runs split on word boundaries; Japanese runs are kept whole and also
    contributed as character bigrams, which is what makes a term like
    "保持期間" match a document that writes "保持" and "期間" separately.
    """
    found: List[str] = []
    for token in _TERM_RE.findall(text.lower()):
        if len(token) < _MIN_TERM_LENGTH:
            continue
        found.append(token)
        if not token.isascii() and len(token) > 2:
            found.extend(token[i : i + 2] for i in range(len(token) - 1))
    return found


def score_caller_document(content: str, terms: List[str]) -> float:
    """Fraction of the question's distinct terms present in the entry, scaled.

    Deterministic and explainable: a caller can see why an entry ranked where it
    did. Returns 0.0 when nothing matches, which keeps unrelated entries out of
    the answer entirely rather than padding it.
    """
    if not terms:
        return 0.0
    haystack = content.lower()
    distinct = set(terms)
    hits = sum(1 for term in distinct if term in haystack)
    if not hits:
        return 0.0
    coverage = hits / len(distinct)
    return round(_CALLER_SCORE_FLOOR + coverage * (_CALLER_SCORE_CEILING - _CALLER_SCORE_FLOOR), 4)


class HybridRetrieveNode(FunctionNode):
    """Retrieve supporting passages for the compliance question.

    Merges two corpora — the caller's validated entries and the built-in
    baseline — scores the caller entries against the question, sorts by
    relevance and caps the result set.

    Output (partial dict — only changed keys):
        retrieval_results, status, error_log.
    """

    # Inner domain node — ANONYMOUS; the outer gate owns the caller check.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error_log: List[str] = list(state.get("error_log") or [])
        legal_contexts: List[Dict[str, Any]] = state.get("legal_contexts") or []
        question_type: str = state.get("question_type", "general")
        parsed_question: str = state.get("parsed_question", "")
        caller_documents: List[Dict[str, Any]] = state.get("caller_documents") or []

        # ── Caller corpus — real retrieval over caller-supplied entries ───────
        terms = _terms(parsed_question)
        caller_results: List[Dict[str, Any]] = []
        for document in caller_documents:
            content = str(document.get("content", ""))
            score = score_caller_document(content, terms)
            if score <= 0.0:
                continue
            caller_results.append(
                {
                    "id": str(document.get("doc_id", "")),
                    "title": f"Caller knowledge base: {document.get('doc_id', '')}",
                    "content": content,
                    "relevance_score": score,
                    "source": "caller",
                }
            )
        caller_results.sort(key=lambda r: r["relevance_score"], reverse=True)

        # ── Baseline corpus for each identified law ──────────────────────────
        baseline_results: List[Dict[str, Any]] = []
        for ctx in legal_contexts:
            law_id = ctx.get("id", "")
            for record in _KB_RECORDS.get(law_id, []):
                baseline_results.append({**record, "source": "baseline"})

        # Always include the general guidance entries.
        seen = {r["id"] for r in baseline_results}
        for record in _KB_RECORDS.get("general", []):
            if record["id"] not in seen:
                baseline_results.append({**record, "source": "baseline"})

        # Reserved caller slots first, then the remainder ranked by relevance.
        reserved = caller_results[:_RESERVED_CALLER_SLOTS]
        remainder = caller_results[_RESERVED_CALLER_SLOTS:] + baseline_results
        remainder.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
        results: List[Dict[str, Any]] = (reserved + remainder)[:_MAX_RESULTS]

        caller_hits = len(caller_results)

        emit_trace_event(
            "hybrid_retrieve_completed",
            {
                "law_count": len(legal_contexts),
                "result_count": len(results),
                "caller_documents_supplied": len(caller_documents),
                "caller_documents_matched": caller_hits,
                "question_type": question_type,
            },
            state,
        )

        return {
            "retrieval_results": results,
            "status": AgentStatus.SUCCESS.value,
            "error_log": error_log,
        }
