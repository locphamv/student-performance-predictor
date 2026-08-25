from app.features import (
    FEATURE_NAMES,
    build_feature_array,
)


def test_feature_names():
    assert FEATURE_NAMES == [
        "study_hours",
        "absences",
        "previous_score",
    ]


def test_build_feature_array():
    features = build_feature_array(
        study_hours=6.0,
        absences=1,
        previous_score=7.5,
    )

    assert features.shape == (1, 3)

    assert features.tolist() == [
        [6.0, 1.0, 7.5]
    ]
