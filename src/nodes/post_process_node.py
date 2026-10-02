"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state, config=None) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus enum constants — never plain strings
#  - Never import from mediator/, api/, or other agents
#
# CMN-C2-659 — PostProcessNode (outer backbone post_process slot)
# Formats the domain result set by VoiceAIComplianceWorkflowGraphNode.merge_output()
# (state["result"]) into state["formatted_output"] for FinalizeNode.
#
# Security notes:
#   Audit: emit_trace_event on every path.
#   Trust: ANONYMOUS — PreProcessNode already screened the caller.

from typing import Any, ClassVar, Dict, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event


class PostProcessNode(FunctionNode):
    """Outer backbone post_process slot for CMN-C2-659.

    Reads state["result"] (populated by VoiceAIComplianceWorkflowGraphNode.merge_output())
    and writes state["formatted_output"] consumed by AgentBaseGraph.get_output()
    as the top-level "output" field in the invoke() response.
    """

    # Outer post_process slot — ANONYMOUS; PreProcessNode already gated the caller.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: str = state.get("result", "")

        emit_trace_event(
            "post_process_completed",
            {"output_length": len(result) if result else 0},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }
