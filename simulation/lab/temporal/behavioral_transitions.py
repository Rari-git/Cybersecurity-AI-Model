"""C3.3 deterministic behavioral-state and transition analysis."""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter
from typing import Any, Mapping

BEHAVIORAL_TRANSITION_SCHEMA = "c3.behavioral-transitions.v1"
SEQUENCE_SCHEMA = "c3.temporal-sequence.v1"
DEFAULT_RAPID_INTERVAL_SECONDS = 0.1
DEFAULT_NORMAL_INTERVAL_SECONDS = 1.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_FIELDS = {"sequence_index","source_step","relative_interval_sec","cumulative_elapsed_sec","time_basis","absolute_timestamp_available","device_id","topic","operation","dropped","ground_truth_label","payload_len","payload_sha256"}
_TIME_SEMANTICS = {"clock_type":"relative_interval","interval_field":"delta_t_sec","cumulative_origin":"zero_at_start_of_recorded_sequence","absolute_timestamps_synthesized":False,"first_frame_interval_included":True}

class BehavioralTransitionError(ValueError):
    """Invalid or unsupported temporal sequence."""

def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
    except (TypeError,ValueError) as exc:
        raise BehavioralTransitionError("result contains non-canonical values") from exc

def _digest(value: Mapping[str,Any],field:str)->str:
    body=dict(value); body.pop(field,None)
    return hashlib.sha256(_canonical(body)).hexdigest()

def _number(value:Any,field:str)->float:
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise BehavioralTransitionError(f"{field} must be numeric")
    result=float(value)
    if not math.isfinite(result) or result<0:
        raise BehavioralTransitionError(f"{field} must be finite and non-negative")
    return result

def _integer(value:Any,field:str,minimum:int=0)->int:
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise BehavioralTransitionError(f"{field} must be an integer >= {minimum}")
    return value

def _validate(sequence:Mapping[str,Any]):
    if not isinstance(sequence,Mapping) or sequence.get("schema_version")!=SEQUENCE_SCHEMA:
        raise BehavioralTransitionError("unsupported temporal sequence schema")
    supplied=sequence.get("sequence_sha256")
    if not isinstance(supplied,str) or not _SHA256_RE.fullmatch(supplied) or _digest(sequence,"sequence_sha256")!=supplied:
        raise BehavioralTransitionError("sequence digest verification failed")
    source=sequence.get("source")
    if not isinstance(source,Mapping):
        raise BehavioralTransitionError("sequence source provenance is missing")
    if source.get("source_type")!="c2_verified_generated_evidence" or source.get("category")!="attack" or source.get("corpus_class")!="adversarial":
        raise BehavioralTransitionError("C3.3 accepts adversarial attack sequences only")
    if source.get("split")!="dev":
        raise BehavioralTransitionError("C3.3 accepts development sequences only")
    if source.get("execution_mode")!="offline_frame_generation_only" or source.get("external_execution_performed") is not False:
        raise BehavioralTransitionError("unsupported source execution provenance")
    if not isinstance(sequence.get("time_semantics"),Mapping) or dict(sequence["time_semantics"])!=_TIME_SEMANTICS:
        raise BehavioralTransitionError("unsupported or modified time semantics")
    events=sequence.get("events"); count=_integer(sequence.get("event_count"),"event_count",1)
    if not isinstance(events,list) or len(events)!=count:
        raise BehavioralTransitionError("sequence event count is invalid")
    result=[]; cumulative=0.0
    for index,event in enumerate(events):
        if not isinstance(event,Mapping) or set(event)!=_EVENT_FIELDS:
            raise BehavioralTransitionError(f"event {index} does not match C3.1 event contract")
        if _integer(event["sequence_index"],"sequence_index")!=index or _integer(event["source_step"],"source_step")!=index:
            raise BehavioralTransitionError("event indices and source steps must be contiguous")
        if event["time_basis"]!="accumulated_delta_t_sec" or event["absolute_timestamp_available"] is not False:
            raise BehavioralTransitionError("unsupported event time basis")
        for field in ("device_id","topic","operation"):
            if not isinstance(event[field],str) or not event[field].strip():
                raise BehavioralTransitionError(f"{field} must be a non-empty string")
        if not isinstance(event["dropped"],bool) or event["ground_truth_label"]!="attack":
            raise BehavioralTransitionError("event dropped flag or ground-truth label is invalid")
        _integer(event["payload_len"],"payload_len")
        if not isinstance(event["payload_sha256"],str) or not _SHA256_RE.fullmatch(event["payload_sha256"]):
            raise BehavioralTransitionError("payload digest is malformed")
        cumulative+=_number(event["relative_interval_sec"],"relative_interval_sec")
        observed=_number(event["cumulative_elapsed_sec"],"cumulative_elapsed_sec")
        if abs(observed-round(cumulative,9))>1e-9:
            raise BehavioralTransitionError("cumulative event time is inconsistent")
        result.append(dict(event))
    return result,dict(source)

def _payload_bucket(size:int)->str:
    if size==0:return "empty"
    if size<=16:return "tiny"
    if size<=64:return "small"
    if size<=256:return "medium"
    if size<=1024:return "large"
    return "very_large"

def _interval_class(value:float,rapid:float,normal:float)->str:
    if value==0:return "simultaneous"
    if value<=rapid:return "rapid"
    if value<=normal:return "normal"
    return "delayed"

def _descriptor(event,rapid,normal):
    return {"device_id":event["device_id"],"operation":event["operation"],"topic":event["topic"],
      "payload_size_bucket":_payload_bucket(event["payload_len"]),
      "interval_class":_interval_class(event["relative_interval_sec"],rapid,normal),"dropped":event["dropped"]}

def _state_id(descriptor):
    return "state_"+hashlib.sha256(_canonical(descriptor)).hexdigest()[:20]

def analyze_behavioral_transitions(sequence:Mapping[str,Any],*,rapid_interval_seconds:float=DEFAULT_RAPID_INTERVAL_SECONDS,normal_interval_seconds:float=DEFAULT_NORMAL_INTERVAL_SECONDS)->dict[str,Any]:
    """Create deterministic descriptive states and conditional transition probabilities."""
    rapid=_number(rapid_interval_seconds,"rapid_interval_seconds")
    normal=_number(normal_interval_seconds,"normal_interval_seconds")
    if rapid<=0 or normal<=rapid:
        raise ValueError("thresholds must satisfy 0 < rapid_interval_seconds < normal_interval_seconds")
    events,source=_validate(sequence)
    descriptors={}; counts=Counter(); event_states=[]
    for event in events:
        descriptor=_descriptor(event,rapid,normal); sid=_state_id(descriptor)
        descriptors[sid]=descriptor; counts[sid]+=1
        event_states.append({"sequence_index":event["sequence_index"],"state_id":sid,"relative_interval_sec":event["relative_interval_sec"]})
    edges=Counter(); edge_dimensions={}; outgoing=Counter(); changes=[]
    for previous,current in zip(event_states,event_states[1:]):
        a,b=previous["state_id"],current["state_id"]; edge=(a,b)
        edges[edge]+=1; outgoing[a]+=1
        before,after=descriptors[a],descriptors[b]
        changed=[key for key in before if before[key]!=after[key]]
        edge_dimensions[edge]=changed
        if a!=b:
            changes.append({"sequence_index":current["sequence_index"],"from_state_id":a,"to_state_id":b,
              "changed_dimensions":changed,"changed_dimension_count":len(changed),
              "multi_dimension_change":len(changed)>=2,"relative_interval_sec":current["relative_interval_sec"]})
    transitions=[]
    for (a,b),amount in sorted(edges.items()):
        transitions.append({"from_state_id":a,"to_state_id":b,"count":amount,
          "outgoing_probability":amount/outgoing[a],
          "transition_kind":"self_transition" if a==b else "state_change",
          "changed_dimensions":edge_dimensions[(a,b)]})
    pairs=max(0,len(events)-1); change_count=len(changes)
    result={"schema_version":BEHAVIORAL_TRANSITION_SCHEMA,
      "source":{"source_type":"c3_verified_temporal_sequence","sequence_sha256":sequence["sequence_sha256"],
        "run_id":source.get("run_id"),"scenario_id":source.get("scenario_id"),"family":source.get("family"),
        "variation_id":source.get("variation_id"),"split":source.get("split"),"corpus_class":source.get("corpus_class"),
        "ground_truth":source.get("ground_truth"),"execution_mode":source.get("execution_mode"),"external_execution_performed":False},
      "configuration":{"rapid_interval_seconds":rapid,"normal_interval_seconds":normal,
        "payload_size_boundaries_bytes":[0,16,64,256,1024],
        "state_dimensions":["device_id","operation","topic","payload_size_bucket","interval_class","dropped"],
        "ground_truth_used_for_state_assignment":False},
      "sequence_summary":{"event_count":len(events),"adjacent_pair_count":pairs,"unique_state_count":len(descriptors),
        "state_change_count":change_count,"state_change_rate":change_count/pairs if pairs else 0.0,
        "multi_dimension_change_count":sum(x["multi_dimension_change"] for x in changes),
        "self_transition_count":sum(n for (a,b),n in edges.items() if a==b),"absolute_timestamps_available":False},
      "states":[{"state_id":sid,"event_count":counts[sid],"event_frequency":counts[sid]/len(events),"descriptor":descriptors[sid]} for sid in sorted(descriptors)],
      "event_state_sequence":event_states,"transitions":transitions,"change_points":changes,
      "interpretation":{"analysis_type":"descriptive_behavioral_transition_analysis","maliciousness_inferred":False,
        "ground_truth_is_not_a_behavioral_state_dimension":True,"transition_probabilities":"conditional_on_the_observed_from_state"}}
    result["behavioral_transition_sha256"]=_digest(result,"behavioral_transition_sha256")
    return result
