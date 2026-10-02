"""AgentCore Platform v1.0"""

# CMN-C2-659 — DomainWorkflowGraph (inner graph)
#
# This is the INNER graph of the Cat 2 two-layer nested architecture. It
# encapsulates the full voice-AI compliance and deployment Q&A workflow:
#
#   START → question_parse → legal_context_identify → hybrid_retrieve
#         → appi_voice_annotate → accessibility_check_format → model_recommend
#         → response_compose → security_gate_output → END
#
# Called by VoiceAIComplianceWorkflowGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   Inherits BaseGraph (fully custom topology — no forced backbone)
#   Implements all 7 BaseGraph abstract methods
#   register_nodes() does NOT call super() (abstract in BaseGraph)
#   Does NOT register initialize / finalize (outer backbone concerns)
#   All domain nodes instantiated with NO constructor arguments
#   get_output() designed together with the outer merge_output()
#   All domain nodes declare TrustLevel.ANONYMOUS
#   _extra_initial_state() seeds the validated caller payload (context bridge)
#   No platform-SDK imports (framework/ and shared/ only)
#   Not placed under src/subagents/

from typing import Any, Dict

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from langgraph.graph import END, START
from src.graph.context_bridge import get_caller_payload
from src.nodes.accessibility_check_format_node import AccessibilityCheckFormatNode
from src.nodes.appi_voice_annotate_node import APPIVoiceAnnotateNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.legal_context_identify_node import LegalContextIdentifyNode
from src.nodes.model_recommend_node import ModelRecommendNode
from src.nodes.question_parse_node import QuestionParseNode
from src.nodes.response_compose_node import ResponseComposeNode
from src.nodes.security_gate_output_node import SecurityGateOutputNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for CMN-C2-659.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by VoiceAIComplianceWorkflowGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          → question_parse             (QuestionParseNode)
          → legal_context_identify     (LegalContextIdentifyNode)
          → hybrid_retrieve            (HybridRetrieveNode)
          → appi_voice_annotate        (APPIVoiceAnnotateNode)
          → accessibility_check_format (AccessibilityCheckFormatNode)
          → model_recommend            (ModelRecommendNode)
          → response_compose           (ResponseComposeNode)
          → security_gate_output       (SecurityGateOutputNode)
          → END

    All nodes are FunctionNode subclasses with TrustLevel.ANONYMOUS.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "cmn_c2_659_domain_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The inner graph consumes no mandatory keys: the settings forwarded from
        config.yaml are advisory here, and a missing file degrades to defaults
        rather than refusing to compile.
        """

    # ── Caller payload ────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the validated caller payload into the inner initial state.

        The framework's GraphNode does not forward input_context into a
        subgraph, so this hook and the matching stash in the outer node's
        extract_input() are the whole bridge. Reading it here rather than in a
        node keeps the hand-off at the graph boundary, where the ordering is
        guaranteed: extract_input() runs before invoke(), invoke() calls this.
        """
        profile, documents = get_caller_payload()
        return {"caller_profile": profile, "caller_documents": documents}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 8 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        """
        self._nodes["question_parse"] = QuestionParseNode()
        self._nodes["legal_context_identify"] = LegalContextIdentifyNode()
        self._nodes["hybrid_retrieve"] = HybridRetrieveNode()
        self._nodes["appi_voice_annotate"] = APPIVoiceAnnotateNode()
        self._nodes["accessibility_check_format"] = AccessibilityCheckFormatNode()
        self._nodes["model_recommend"] = ModelRecommendNode()
        self._nodes["response_compose"] = ResponseComposeNode()
        self._nodes["security_gate_output"] = SecurityGateOutputNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear domain topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear — all 8 domain nodes execute in
        sequence regardless of question type, and each one decides its own
        applicability from legal_contexts and question_type.
        """
        self._sg.add_edge(START, "question_parse")
        self._sg.add_edge("question_parse", "legal_context_identify")
        self._sg.add_edge("legal_context_identify", "hybrid_retrieve")
        self._sg.add_edge("hybrid_retrieve", "appi_voice_annotate")
        self._sg.add_edge("appi_voice_annotate", "accessibility_check_format")
        self._sg.add_edge("accessibility_check_format", "model_recommend")
        self._sg.add_edge("model_recommend", "response_compose")
        self._sg.add_edge("response_compose", "security_gate_output")
        self._sg.add_edge("security_gate_output", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: State) -> str:
        """Conditional routing — required by the BaseGraph contract.

        The topology above is linear: add_conditional_edges() is never called,
        so this method has no runtime caller here. It is annotated with THIS
        graph's own State regardless, because that is what a path callable must
        carry the moment one is wired: the graph runtime reads the annotation as
        the callable's input schema and PROJECTS AWAY every field the annotation
        does not declare. A route annotated with the generic base state would be
        handed a state whose domain fields are always absent — the branch would
        never be taken in a real invocation, while unit tests that pass a full
        dict directly would keep passing.

        Returns END on error to prevent unexpected re-entry.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "security_gate_output"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        Received by VoiceAIComplianceWorkflowGraphNode.merge_output().

        A non-SUCCESS run returns NO document. Resolving the document as
        `secured_response or draft_response` without consulting the status is
        precisely how an answer the output boundary refused travels onward:
        draft_response holds the text as the pipeline produced it, before the
        boundary screened it, and the boundary's decision is recorded only in
        the status.
        """
        released = None
        if state.get("status") == AgentStatus.SUCCESS.value:
            released = state.get("secured_response")
        return {
            "output": released,
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
