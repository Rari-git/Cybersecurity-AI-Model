from training.evaluation_pep import SimulatedPep


def test_simulated_pep_has_explicit_failure_and_timeout_cases():
    pep = SimulatedPep(fail_decisions={"fail"}, timeout_decisions={"timeout"}, simulated_latency_ms=0)
    assert pep.apply("gas_sensor_01", True, "pass") is True
    assert pep.apply("gas_sensor_01", True, "fail") is False
    assert pep.apply("gas_sensor_01", True, "timeout") is False
    assert [x[1] for x in pep.history] == [True, False, False]
