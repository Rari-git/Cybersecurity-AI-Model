"""C3 temporal analysis components."""

from simulation.lab.temporal.feature_extractor import (
    TEMPORAL_FEATURE_SCHEMA,
    TemporalFeatureError,
    extract_temporal_features,
)

from simulation.lab.temporal.behavioral_transitions import (
    BEHAVIORAL_TRANSITION_SCHEMA,
    BehavioralTransitionError,
    analyze_behavioral_transitions,
)

from simulation.lab.temporal.transition_profile import (
    CLEAN_TRANSITION_PROFILE_SCHEMA,
    TransitionProfileError,
    build_clean_transition_profile,
    evaluate_clean_transition_profile,
)

__all__ = [
    "TEMPORAL_FEATURE_SCHEMA",
    "TemporalFeatureError",
    "extract_temporal_features",
    "BEHAVIORAL_TRANSITION_SCHEMA",
    "BehavioralTransitionError",
    "analyze_behavioral_transitions",
    "CLEAN_TRANSITION_PROFILE_SCHEMA",
    "TransitionProfileError",
    "build_clean_transition_profile",
    "evaluate_clean_transition_profile",
]
