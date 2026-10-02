# Unit tests for the eight inner workflow nodes and the two boundary helpers.
#
# The evaluation nodes are tested on their arithmetic — the number the answer
# reports has to be the number the threshold implies — and the output boundary
# on containment, which is what an error status alone does not give you.

import pytest
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from src.nodes.accessibility_check_format_node import AccessibilityCheckFormatNode
from src.nodes.appi_voice_annotate_node import APPIVoiceAnnotateNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode, score_caller_document
from src.nodes.legal_context_identify_node import LegalContextIdentifyNode
from src.nodes.model_recommend_node import ModelRecommendNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.question_parse_node import QuestionParseNode
from src.nodes.response_compose_node import ResponseComposeNode
from src.nodes.security_gate_output_node import (
    WITHHELD_NOTICE,
    SecurityGateOutputNode,
    _security_gate_output,
    check_labels,
    scan_value,
)

ALL_NODES = [
    QuestionParseNode,
    LegalContextIdentifyNode,
    HybridRetrieveNode,
    APPIVoiceAnnotateNode,
    AccessibilityCheckFormatNode,
    ModelRecommendNode,
    ResponseComposeNode,
    SecurityGateOutputNode,
    PostProcessNode,
]


@pytest.fixture(autouse=True)
def patch_emit(monkeypatch):
    """Silence the audit sink on every node module used below."""
    for module in [
        "question_parse_node",
        "legal_context_identify_node",
        "hybrid_retrieve_node",
        "appi_voice_annotate_node",
        "accessibility_check_format_node",
        "model_recommend_node",
        "response_compose_node",
        "security_gate_output_node",
        "post_process_node",
    ]:
        monkeypatch.setattr(f"src.nodes.{module}.emit_trace_event", lambda *a, **k: None)


def state(**overrides):
    base = {"node_history": [], "error_log": [], "input_context": {}}
    base.update(overrides)
    return base


class TestNodeContracts:
    @pytest.mark.parametrize("node_cls", ALL_NODES)
    def test_inner_nodes_declare_a_trust_level(self, node_cls):
        assert node_cls.required_trust_level == TrustLevel.ANONYMOUS

    @pytest.mark.parametrize("node_cls", ALL_NODES)
    def test_execute_takes_state_first(self, node_cls):
        import inspect

        params = list(inspect.signature(node_cls.execute).parameters)
        assert params[:2] == ["self", "state"]


class TestQuestionParse:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("音声録音の同意と保持期間について", "appi"),
            ("バリアフリー法のアクセシビリティ要件", "accessibility"),
            ("コンプライアンス上の義務は", "legal_compliance"),
            ("デプロイメントのlatency設定", "technical_config"),
            ("こんにちは", "general"),
        ],
    )
    def test_question_type_classification(self, question, expected):
        result = QuestionParseNode().execute(state(validated_input=question))
        assert result["question_type"] == expected

    def test_whitespace_is_collapsed(self):
        result = QuestionParseNode().execute(state(validated_input="音声  AI   の\n要件"))
        assert result["parsed_question"] == "音声 AI の 要件"

    def test_absent_input_is_an_error(self):
        result = QuestionParseNode().execute(state(validated_input=""))
        assert result["status"] == AgentStatus.ERROR.value


class TestLegalContextIdentify:
    def test_recording_question_matches_the_recording_rules(self):
        result = LegalContextIdentifyNode().execute(
            state(parsed_question="音声録音の同意と保持期間", question_type="appi")
        )
        assert "APPI_2026" in {ctx["id"] for ctx in result["legal_contexts"]}

    def test_no_match_is_still_a_success(self):
        result = LegalContextIdentifyNode().execute(state(parsed_question="hello", question_type="general"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["legal_contexts"] == []

    def test_absent_question_is_an_error(self):
        result = LegalContextIdentifyNode().execute(state(parsed_question=""))
        assert result["status"] == AgentStatus.ERROR.value


class TestHybridRetrieve:
    def test_no_caller_corpus_degrades_to_the_baseline(self):
        result = HybridRetrieveNode().execute(
            state(parsed_question="保持期間", question_type="appi", legal_contexts=[{"id": "APPI_2026"}])
        )
        assert result["retrieval_results"]
        assert all(r["source"] == "baseline" for r in result["retrieval_results"])

    def test_a_matching_caller_entry_reaches_the_answer(self):
        documents = [{"doc_id": "policy_01", "content": "当社の保持期間は180日です"}]
        result = HybridRetrieveNode().execute(
            state(
                parsed_question="保持期間について",
                question_type="appi",
                legal_contexts=[{"id": "APPI_2026"}],
                caller_documents=documents,
            )
        )
        ids = [r["id"] for r in result["retrieval_results"]]
        assert "policy_01" in ids, "a caller entry that matches must not be crowded out by the baseline"

    def test_an_unrelated_caller_entry_is_left_out(self):
        documents = [{"doc_id": "unrelated", "content": "zzz qqq"}]
        result = HybridRetrieveNode().execute(
            state(parsed_question="保持期間について", question_type="appi", caller_documents=documents)
        )
        assert "unrelated" not in [r["id"] for r in result["retrieval_results"]]

    def test_scoring_rises_with_term_coverage(self):
        terms = ["retention", "period", "deletion"]
        one = score_caller_document("retention rules", terms)
        three = score_caller_document("retention period and deletion rules", terms)
        assert 0.0 < one < three <= 0.94

    def test_scoring_is_zero_without_terms(self):
        assert score_caller_document("anything", []) == 0.0


class TestRecordingGovernanceEvaluation:
    def test_no_profile_yields_no_findings(self):
        result = APPIVoiceAnnotateNode().execute(state(question_type="appi", legal_contexts=[]))
        assert result["compliance_findings"] == []

    def test_retention_overrun_is_reported_with_the_gap(self):
        result = APPIVoiceAnnotateNode().execute(
            state(question_type="appi", caller_profile={"recording_retention_days": 180.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "retention_exceeds_default")
        assert finding["severity"] == "action_required"
        assert "90 days beyond" in finding["detail"]

    def test_a_small_overrun_is_attention_not_action(self):
        result = APPIVoiceAnnotateNode().execute(
            state(question_type="appi", caller_profile={"recording_retention_days": 100.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "retention_exceeds_default")
        assert finding["severity"] == "attention"

    def test_a_compliant_retention_period_is_reported_as_info(self):
        result = APPIVoiceAnnotateNode().execute(
            state(question_type="appi", caller_profile={"recording_retention_days": 30.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "retention_within_default")
        assert finding["severity"] == "info"

    def test_deletion_service_level_overrun_is_reported(self):
        result = APPIVoiceAnnotateNode().execute(
            state(question_type="appi", caller_profile={"deletion_sla_days": 45.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "deletion_sla_exceeds_limit")
        assert "15 beyond" in finding["detail"]

    def test_prior_findings_are_preserved(self):
        prior = [{"area": "x", "code": "y", "severity": "info", "detail": "d"}]
        result = APPIVoiceAnnotateNode().execute(
            state(question_type="appi", compliance_findings=prior, caller_profile={"deletion_sla_days": 1.0})
        )
        assert result["compliance_findings"][0] == prior[0]


class TestAccessibilityEvaluation:
    def test_a_narrow_rate_range_is_reported_at_both_ends(self):
        result = AccessibilityCheckFormatNode().execute(
            state(question_type="accessibility", caller_profile={"speech_rate_min": 0.75, "speech_rate_max": 1.5})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "speech_rate_range_insufficient")
        assert "0.75x" in finding["detail"] and "1.50x" in finding["detail"]

    def test_a_sufficient_rate_range_is_info(self):
        result = AccessibilityCheckFormatNode().execute(
            state(question_type="accessibility", caller_profile={"speech_rate_min": 0.5, "speech_rate_max": 2.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "speech_rate_range_sufficient")
        assert finding["severity"] == "info"


class TestModelEvaluation:
    def test_accuracy_shortfall_is_reported_in_percentage_points(self):
        result = ModelRecommendNode().execute(state(question_type="general", caller_profile={"intent_accuracy": 0.81}))
        finding = next(f for f in result["compliance_findings"] if f["code"] == "intent_accuracy_below_target")
        assert "81.0%" in finding["detail"]
        assert "4.0 percentage points" in finding["detail"]

    def test_latency_overrun_is_reported_in_milliseconds(self):
        result = ModelRecommendNode().execute(
            state(question_type="general", caller_profile={"response_latency_ms": 450.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "latency_above_target")
        assert "150ms above" in finding["detail"]
        assert finding["severity"] == "attention"

    def test_a_large_latency_overrun_escalates(self):
        result = ModelRecommendNode().execute(
            state(question_type="general", caller_profile={"response_latency_ms": 900.0})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "latency_above_target")
        assert finding["severity"] == "action_required"

    def test_missing_languages_are_named(self):
        result = ModelRecommendNode().execute(
            state(question_type="general", caller_profile={"supported_languages": ["japanese", "mandarin"]})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "language_coverage_incomplete")
        assert "vietnamese" in finding["detail"]
        assert "missing 3 of the 5" in finding["detail"]

    def test_full_coverage_is_info(self):
        languages = ["japanese", "mandarin", "vietnamese", "indonesian", "tagalog"]
        result = ModelRecommendNode().execute(
            state(question_type="general", caller_profile={"supported_languages": languages})
        )
        finding = next(f for f in result["compliance_findings"] if f["code"] == "language_coverage_complete")
        assert finding["severity"] == "info"


class TestResponseCompose:
    def test_overall_severity_is_the_strongest_finding(self):
        findings = [
            {"area": "a", "code": "a", "severity": "info", "detail": "d"},
            {"area": "b", "code": "b", "severity": "action_required", "detail": "d"},
            {"area": "c", "code": "c", "severity": "attention", "detail": "d"},
        ]
        result = ResponseComposeNode().execute(state(parsed_question="q", compliance_findings=findings))
        assert result["overall_severity"] == "action_required"
        assert "Overall assessment: action_required" in result["draft_response"]

    def test_no_findings_yields_info_and_says_why(self):
        result = ResponseComposeNode().execute(state(parsed_question="q"))
        assert result["overall_severity"] == "info"
        assert "No deployment profile was supplied" in result["draft_response"]

    def test_the_document_states_its_own_rendering_rules(self):
        result = ResponseComposeNode().execute(state(parsed_question="q"))
        assert "Output schema:" in result["draft_response"]

    def test_absent_question_is_an_error(self):
        result = ResponseComposeNode().execute(state(parsed_question=""))
        assert result["status"] == AgentStatus.ERROR.value


class TestOutputBoundary:
    def test_clean_text_is_released(self):
        result = SecurityGateOutputNode().execute(state(draft_response="Voice AI guidance. 保持期間は90日。"))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["secured_response"].startswith("Voice AI guidance")

    @pytest.mark.parametrize(
        "text",
        [
            "report api_key: abcd1234efgh5678",
            "password = hunter2",
            "Authorization: Bearer abcdefghijklmnopqrstu",
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghij",
            "AKIAIOSFODNN7EXAMPLE",
            "sk-abcdefghijklmnopqrstuvwxyz012345",
        ],
    )
    def test_credential_shapes_are_withheld(self, text):
        result = SecurityGateOutputNode().execute(state(draft_response=f"guidance\n{text}\n"))
        assert result["status"] == AgentStatus.ERROR.value

    @pytest.mark.parametrize(
        "text",
        ["contact admin@company.co.jp", "call 090-1234-5678", "card 4111111111111111"],
    )
    def test_personal_data_is_withheld(self, text):
        result = SecurityGateOutputNode().execute(state(draft_response=f"guidance\n{text}\n"))
        assert result["status"] == AgentStatus.ERROR.value

    def test_a_withheld_document_is_replaced_not_merely_flagged(self):
        # An error status alone is not containment: a node returns a PARTIAL
        # update, so a field the update omits keeps the value it had, and the
        # callers that resolve `secured_response or draft_response` read it.
        secret = "abcd1234efgh5678"
        result = SecurityGateOutputNode().execute(state(draft_response=f"guidance api_key: {secret}"))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["draft_response"] == WITHHELD_NOTICE
        assert result["secured_response"] == WITHHELD_NOTICE
        assert secret not in str(result)

    def test_the_replacement_is_truthy(self):
        # A falsy placeholder is not a cleared field — it re-activates exactly
        # the `or` fallback the clearing exists to defeat.
        assert bool(WITHHELD_NOTICE)

    def test_the_message_names_the_class_and_quotes_nothing(self):
        secret = "abcd1234efgh5678"
        result = SecurityGateOutputNode().execute(state(draft_response=f"api_key: {secret}"))
        message = " ".join(result["error_log"])
        assert "withheld" in message
        assert secret not in message

    def test_a_credential_nested_in_the_sources_is_caught(self):
        # Caller text arrives inside lists of mappings and only part of it is
        # quoted into the document; a scan of top-level strings alone reports
        # nothing for a credential one level down.
        sources = [{"id": "d1", "content": "harmless"}, {"id": "d2", "content": "api_key: abcd1234efgh5678"}]
        result = SecurityGateOutputNode().execute(state(draft_response="clean guidance", retrieval_results=sources))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["secured_response"] == WITHHELD_NOTICE

    def test_the_nested_scan_control_passes_on_clean_sources(self):
        # The nested probe alone cannot tell "scan is blind" from "probe is
        # wrong"; this is the control that proves it looks at all.
        sources = [{"id": "d1", "content": "harmless"}, {"id": "d2", "content": "also harmless"}]
        assert scan_value(sources) is None
        result = SecurityGateOutputNode().execute(state(draft_response="clean guidance", retrieval_results=sources))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_a_credential_in_a_nested_key_is_caught(self):
        assert scan_value({"api_key: abcd1234efgh5678": "value"}) is not None

    def test_absent_document_is_an_error_and_clears_the_fields(self):
        result = SecurityGateOutputNode().execute(state(draft_response=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["secured_response"] == WITHHELD_NOTICE

    def test_the_boundary_helper_is_a_module_function_not_an_override(self):
        # The framework's gate of the same name is final on FunctionNode; a
        # subclass attribute would be rejected at class definition time.
        assert "_security_gate_output" not in SecurityGateOutputNode.__dict__
        assert _security_gate_output("clean") is None

    def test_labels_are_revalidated_at_the_boundary(self):
        assert check_labels([{"doc_id": "policy_01"}], {}) is None
        assert check_labels([{"doc_id": "Policy 01"}], {}) is not None
        assert check_labels([], {"supported_languages": ["Japanese"]}) is not None
        assert check_labels([], {"supported_languages": ["japanese"]}) is None

    def test_a_label_violation_withholds_the_document(self):
        result = SecurityGateOutputNode().execute(
            state(draft_response="clean guidance", caller_documents=[{"doc_id": "Not Inert"}])
        )
        assert result["status"] == AgentStatus.ERROR.value
        assert result["secured_response"] == WITHHELD_NOTICE

    def test_structural_numbers_survive_unchanged(self):
        # This agent renders no monetary aggregates, so there is no numeric snap
        # in the pipeline — day counts, millisecond budgets, years and version
        # strings reach the caller byte-identical.
        text = "retention 90 days, latency 300ms, 2026 amendment, rate 0.5x-2.0x, WER 8.512345%"
        result = SecurityGateOutputNode().execute(state(draft_response=text))
        assert result["secured_response"] == text


class TestPostProcess:
    def test_result_is_forwarded_as_formatted_output(self):
        result = PostProcessNode().execute(state(result="the document"))
        assert result["formatted_output"] == "the document"

    def test_absent_result_is_tolerated(self):
        result = PostProcessNode().execute(state())
        assert result["status"] == AgentStatus.SUCCESS.value
