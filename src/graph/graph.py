"""AgentCore Platform v1.0"""

# CMN-C2-659 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max_retry)
#                                          pre_process
#
#   The `main` slot is a GraphNode subclass (VoiceAIComplianceWorkflowGraphNode)
#   that delegates the full domain workflow to DomainWorkflowGraph (inner graph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step topology)
#   src/graph/context_bridge.py        ← carries validated caller data inward
#
# Rules enforced:
#   VoiceAIComplianceDeploymentQAAgent inherits AgentBaseGraph
#   class name matches config/agent.yaml `class:` AND src/api/server.py import
#   super().register_nodes() called first (fills initialize + finalize)
#   VoiceAIComplianceWorkflowGraphNode assigned to self._nodes["main"]
#   merge_output() returns only changed keys
#   GraphNode (main slot) declared ANONYMOUS — the outer pre_process gate owns
#     the caller trust check
#   add_edges() NOT overridden on the outer graph
#   No platform-SDK imports (framework/ and shared/ only)

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, cast

import yaml
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.graph.context_bridge import set_caller_payload
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and hands to the graph
    constructor. The standalone server reads it through this function so a
    registry-loaded agent and a directly deployed one see identical settings —
    without it, every value declared in config.yaml is inert in one of the two
    deployments and the difference is invisible: nothing fails, the framework
    simply falls back to its own defaults.

    Returns an empty mapping — never raises — when the file is absent or does
    not parse, so a missing file degrades to framework defaults rather than
    breaking start-up.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        loaded = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


class VoiceAIComplianceWorkflowGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (the inner Cat 2 graph).
    Called by the backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()   — instantiate and return DomainWorkflowGraph
      extract_input()  — pass the validated question into the inner invoke, and
                         stash the validated caller payload on the context bridge
      merge_output()   — map sub_result fields into the outer state delta
      error_strategy   — "propagate": re-raise inner errors (fail fast)

    Trust: ANONYMOUS — the outer PreProcessNode (VERIFIED_EXTERNAL) already
    performed the caller check. Inner graph nodes inherit the caller's
    InvocationContext unchanged.
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (fail fast).
    error_strategy: ClassVar[str] = "propagate"

    # False: interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    # Inner boundary trust level — ANONYMOUS so any outer-authenticated caller
    # can proceed.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        Lazy import avoids circular-import risk at module load time. The runtime
        settings are forwarded so a value declared in config.yaml is live in the
        inner graph as well as the outer one.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the declared runtime settings to the inner graph.

        Reads the same config.yaml the outer graph was constructed from. Kept as
        its own method so the inner graph's configuration has one documented
        source rather than being reconstructed at each call site.
        """
        return runtime_config()

    def extract_input(self, state: AgentState) -> str:
        """Return the question string passed into inner_graph.invoke().

        Also stashes the VALIDATED caller payload on the context bridge. The
        framework's GraphNode does not forward input_context into the inner
        graph, so without this the deployment profile and the caller's
        knowledge-base entries would be absent from every inner node and the
        agent would silently answer from its built-in baseline alone.

        What crosses is what PreProcessNode produced, never the raw context:
        caller data is bounded exactly once, upstream of this point.
        """
        set_caller_payload(
            cast(Optional[Dict[str, Any]], state.get("caller_profile")),
            cast(Optional[List[Dict[str, Any]]], state.get("caller_documents")),
        )
        return cast(str, state.get("validated_input", state.get("user_input", "")))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        Returns ONLY changed keys — never the full state.

        Key coupling (designed with DomainWorkflowGraph.get_output()):
            Inner get_output() emits:  "output", "status", "trace_id",
                                        "correlation_id", "node_history"
            This merge_output() reads: sub_result.get("output"),
                                        sub_result.get("status")

        A non-SUCCESS inner run contributes NO document. The inner graph already
        withholds it, but repeating the check here means neither layer alone has
        to be right for an unreleased answer to stay unreleased.
        """
        if sub_result.get("status") != AgentStatus.SUCCESS.value:
            return {"result": None, "status": sub_result.get("status")}
        return {
            "result": sub_result.get("output"),
            "status": sub_result.get("status"),
        }


class VoiceAIComplianceDeploymentQAAgent(AgentBaseGraph):
    """Outer graph for CMN-C2-659 (Cat 2).

    Inherits AgentBaseGraph directly (L1 Base). Domain logic is fully
    encapsulated in VoiceAIComplianceWorkflowGraphNode (main slot), which
    delegates to DomainWorkflowGraph (inner graph).

    Class name: VoiceAIComplianceDeploymentQAAgent
    MUST match:
      config/agent.yaml   `class: "src.graph.graph.VoiceAIComplianceDeploymentQAAgent"`
      src/api/server.py   `from src.graph.graph import VoiceAIComplianceDeploymentQAAgent`

    Construct with the runtime settings — `Agent(config=runtime_config())` —
    exactly as the platform registry does. AgentBaseGraph consumes `max_retry`
    from that mapping for retry routing, so the declared value is live in both
    deployments.

    Backbone (fixed):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process: PreProcessNode (caller trust gate + full input contract)
      - main:        VoiceAIComplianceWorkflowGraphNode (inner workflow)
      - post_process: PostProcessNode (output formatting)

    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier — matches config/agent.yaml `name:` field."""
        return "VoiceAIComplianceDeploymentQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (schema_version, session_id, trust
        level) and finalize node (response metadata, total time).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = VoiceAIComplianceWorkflowGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Project the final state into the caller-facing response.

        The inherited accessor resolves the caller-facing document as
        `formatted_output or result`, with no reference to the run's status. On
        a run the output boundary refused, those fields are exactly where the
        unreleased answer sits — so the inherited expression publishes the
        withheld document inside an envelope whose status says the run failed.

        Only a SUCCESS run therefore yields a document here; every other
        terminal status yields none, whatever the state fields still hold. This
        is deliberately independent of the boundary node clearing them: either
        measure contains the leak alone, and together neither a new failure path
        that forgets to clear nor a future edit to this accessor can put an
        unscreened document in front of a caller.
        """
        base: Dict[str, Any] = dict(super().get_output(state))
        if state.get("status") != AgentStatus.SUCCESS.value:
            base["output"] = None
        return base

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Back-compat alias — callers that reference Graph still work.
Graph = VoiceAIComplianceDeploymentQAAgent
