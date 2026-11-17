from __future__ import annotations
import copy,hashlib,json
import pytest
from simulation.lab.temporal.behavioral_transitions import BEHAVIORAL_TRANSITION_SCHEMA,BehavioralTransitionError,analyze_behavioral_transitions

def _sequence(specs):
    events=[]; elapsed=0.0
    for i,(device,operation,topic,size,interval,dropped) in enumerate(specs):
        elapsed+=interval
        events.append({"sequence_index":i,"source_step":i,"relative_interval_sec":interval,"cumulative_elapsed_sec":round(elapsed,9),
          "time_basis":"accumulated_delta_t_sec","absolute_timestamp_available":False,"device_id":device,"topic":topic,"operation":operation,
          "dropped":dropped,"ground_truth_label":"attack","payload_len":size,"payload_sha256":hashlib.sha256(f"payload-{i}".encode()).hexdigest()})
    value={"schema_version":"c3.temporal-sequence.v1",
      "source":{"source_type":"c2_verified_generated_evidence","run_id":"run-test","scenario_id":"scenario-test","family":"test-family",
        "category":"attack","corpus_class":"adversarial","split":"dev","ground_truth":"attack","variation_id":"variation-test",
        "execution_mode":"offline_frame_generation_only","external_execution_performed":False},
      "time_semantics":{"clock_type":"relative_interval","interval_field":"delta_t_sec","cumulative_origin":"zero_at_start_of_recorded_sequence",
        "absolute_timestamps_synthesized":False,"first_frame_interval_included":True},"event_count":len(events),"events":events}
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    value["sequence_sha256"]=hashlib.sha256(raw).hexdigest()
    return value

def _rehash(value):
    raw=json.dumps({k:v for k,v in value.items() if k!="sequence_sha256"},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    value["sequence_sha256"]=hashlib.sha256(raw).hexdigest()
    return value

@pytest.fixture
def repeated_sequence():
    return _sequence([("sensor-1","publish","sensors/temp",8,0.05,False),("sensor-1","publish","sensors/temp",8,0.05,False),
      ("sensor-1","publish","sensors/temp",8,0.05,False),("sensor-1","read","sensors/status",32,0.05,False),
      ("sensor-1","read","sensors/status",32,0.05,False)])

def test_schema_provenance_and_descriptive_only(repeated_sequence):
    result=analyze_behavioral_transitions(repeated_sequence)
    assert result["schema_version"]==BEHAVIORAL_TRANSITION_SCHEMA
    assert result["source"]["sequence_sha256"]==repeated_sequence["sequence_sha256"]
    assert result["source"]["split"]=="dev" and result["source"]["external_execution_performed"] is False
    assert result["interpretation"]["maliciousness_inferred"] is False

def test_deterministic_output(repeated_sequence):
    assert analyze_behavioral_transitions(repeated_sequence)==analyze_behavioral_transitions(repeated_sequence)

def test_self_transitions_are_counted(repeated_sequence):
    result=analyze_behavioral_transitions(repeated_sequence)
    assert result["sequence_summary"]["self_transition_count"]==3
    assert any(x["transition_kind"]=="self_transition" for x in result["transitions"])

def test_conditional_probabilities_sum_to_one(repeated_sequence):
    result=analyze_behavioral_transitions(repeated_sequence); totals={}
    for edge in result["transitions"]: totals[edge["from_state_id"]]=totals.get(edge["from_state_id"],0)+edge["outgoing_probability"]
    assert all(x==pytest.approx(1.0) for x in totals.values())

def test_change_points_explain_changed_dimensions(repeated_sequence):
    change=analyze_behavioral_transitions(repeated_sequence)["change_points"][0]
    assert set(change["changed_dimensions"])=={"operation","topic","payload_size_bucket"}
    assert change["multi_dimension_change"] is True

@pytest.mark.parametrize("size,expected",[(0,"empty"),(1,"tiny"),(16,"tiny"),(17,"small"),(64,"small"),(65,"medium"),(256,"medium"),(257,"large"),(1024,"large"),(1025,"very_large")])
def test_payload_size_buckets(size,expected):
    assert analyze_behavioral_transitions(_sequence([("d","op","t",size,0.0,False)]))["states"][0]["descriptor"]["payload_size_bucket"]==expected

@pytest.mark.parametrize("interval,expected",[(0.0,"simultaneous"),(0.1,"rapid"),(0.100001,"normal"),(1.0,"normal"),(1.000001,"delayed")])
def test_interval_classes(interval,expected):
    assert analyze_behavioral_transitions(_sequence([("d","op","t",1,interval,False)]))["states"][0]["descriptor"]["interval_class"]==expected

def test_ground_truth_metadata_does_not_define_state(repeated_sequence):
    first=analyze_behavioral_transitions(repeated_sequence); changed=copy.deepcopy(repeated_sequence)
    changed["source"]["ground_truth"]="other-metadata"; _rehash(changed)
    second=analyze_behavioral_transitions(changed)
    assert [x["state_id"] for x in first["event_state_sequence"]]==[x["state_id"] for x in second["event_state_sequence"]]

def test_dropped_status_is_state_dimension():
    result=analyze_behavioral_transitions(_sequence([("d","op","t",1,0.0,False),("d","op","t",1,0.0,True)]))
    assert result["sequence_summary"]["unique_state_count"]==2
    assert result["change_points"][0]["changed_dimensions"]==["dropped"]

def test_single_event_has_no_transitions():
    result=analyze_behavioral_transitions(_sequence([("d","op","t",1,0.0,False)]))
    assert result["transitions"]==[] and result["sequence_summary"]["state_change_rate"]==0.0

def test_tampered_sequence_rejected(repeated_sequence):
    changed=copy.deepcopy(repeated_sequence); changed["events"][0]["topic"]="tampered"
    with pytest.raises(BehavioralTransitionError,match="digest"): analyze_behavioral_transitions(changed)

@pytest.mark.parametrize("split",["held_out","regression","train"])
def test_non_dev_splits_rejected(repeated_sequence,split):
    changed=copy.deepcopy(repeated_sequence); changed["source"]["split"]=split; _rehash(changed)
    with pytest.raises(BehavioralTransitionError,match="development"): analyze_behavioral_transitions(changed)

def test_invalid_threshold_order_rejected(repeated_sequence):
    with pytest.raises(ValueError,match="thresholds"): analyze_behavioral_transitions(repeated_sequence,rapid_interval_seconds=1.0,normal_interval_seconds=0.5)

def test_boolean_interval_rejected(repeated_sequence):
    changed=copy.deepcopy(repeated_sequence); changed["events"][0]["relative_interval_sec"]=True; _rehash(changed)
    with pytest.raises(BehavioralTransitionError,match="relative_interval_sec"): analyze_behavioral_transitions(changed)
