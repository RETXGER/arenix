from arenix_engine import AttackerRole, MockAdapter


def _attacker(profile: str) -> AttackerRole:
    return AttackerRole(MockAdapter("mock", "mock-attacker"), profile=profile)


def test_attacker_profile_soft_uses_low_pressure_strategy():
    attacker = _attacker("soft")
    strategy = attacker._turn_strategy_instruction(turn_number=1, last_turn_analysis=None).lower()
    assert "low-pressure" in strategy


def test_attacker_profile_compliance_uses_audit_language():
    attacker = _attacker("compliance")
    strategy = attacker._turn_strategy_instruction(turn_number=1, last_turn_analysis=None).lower()
    assert "audit" in strategy or "governance" in strategy


def test_attacker_profile_aggressive_keeps_escalation_fallback():
    attacker = _attacker("aggressive")
    prompt = attacker._fallback_aggressive_prompt(2).lower()
    assert "unrestricted audit mode" in prompt
