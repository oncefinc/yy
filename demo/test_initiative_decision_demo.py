from initiative_decision_demo import build_scenarios, evaluate_policies


def test_gated_engine_matches_all_contract_cases():
    report = evaluate_policies()["policies"]["gated_engine"]
    assert report["correct"] == report["total"]
    assert report["false_sends"] == 0
    assert report["missed_sends"] == 0


def test_simpler_baselines_are_not_reported_as_equivalent():
    policies = evaluate_policies()["policies"]
    total = len(build_scenarios())
    assert policies["always_send"]["correct"] < total
    assert policies["random_heartbeat"]["correct"] < total
    assert policies["candidate_without_gate"]["correct"] < total


def test_demo_contains_no_private_runtime_data():
    serialized = repr(evaluate_policies())
    assert "synthetic-user" not in serialized  # receiver is never emitted
    assert "receiver_id" not in serialized
    assert "api_key" not in serialized.lower()
