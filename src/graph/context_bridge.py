"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the caller's validated data across the
# outer→inner graph boundary.
#
# Why this exists: the framework's GraphNode invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does NOT forward
# the outer state's input_context. Without a bridge, every inner-node read of
# the caller's deployment profile or knowledge-base entries would see nothing
# at all, through the whole nested graph, while unit tests that hand a node a
# populated dict keep passing. The sanctioned subclass hooks bridge it:
#
#   VoiceAIComplianceWorkflowGraphNode.extract_input(state)
#       [runs BEFORE subgraph.invoke]  → set_caller_payload(...)
#   DomainWorkflowGraph._extra_initial_state()
#       [runs INSIDE subgraph.invoke]  → returns the stashed payload
#
# What crosses is the VALIDATED payload the outer pre_process node produced —
# never the raw input_context. Raw caller data has exactly one gate in this
# agent, and it is upstream of this bridge.
#
# A ContextVar keeps the hand-off correct per thread and per task, so
# concurrent invocations in one process cannot see each other's caller data.

from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

_CALLER_PROFILE: ContextVar[Optional[Dict[str, Any]]] = ContextVar("cmn_c2_659_caller_profile", default=None)
_CALLER_DOCUMENTS: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("cmn_c2_659_caller_documents", default=None)


def set_caller_payload(
    profile: Optional[Dict[str, Any]],
    documents: Optional[List[Dict[str, Any]]],
) -> None:
    """Stash the validated caller payload for the imminent inner-graph invoke."""
    _CALLER_PROFILE.set(dict(profile) if profile else {})
    _CALLER_DOCUMENTS.set([dict(doc) for doc in documents] if documents else [])


def get_caller_payload() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read (without consuming) the stashed payload; empties when none was set."""
    return _CALLER_PROFILE.get() or {}, _CALLER_DOCUMENTS.get() or []


def clear_caller_payload() -> None:
    """Drop the stashed payload. Used by tests to assert the absent-data path."""
    _CALLER_PROFILE.set(None)
    _CALLER_DOCUMENTS.set(None)
