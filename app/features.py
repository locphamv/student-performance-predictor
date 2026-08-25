import numpy as np


FEATURE_NAMES = [
    "study_hours",
    "absences",
    "previous_score",
]


def build_feature_array(
    study_hours: float,
    absences: int,
    previous_score: float,
) -> np.ndarray:
    feature_values = {
        "study_hours": study_hours,
        "absences": absences,
        "previous_score": previous_score,
    }

    return np.array([
        [
            feature_values[name]
            for name in FEATURE_NAMES
        ]
    ])
