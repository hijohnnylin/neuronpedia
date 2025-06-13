from typing import Any

import pandas as pd
import pytest
from neuronpedia_autointerp_client.models.score_embedding_post200_response_breakdown_inner import (
    ScoreEmbeddingPost200ResponseBreakdownInner,
)
from neuronpedia_autointerp_client.models.score_fuzz_detection_post200_response_breakdown_inner import (
    ScoreFuzzDetectionPost200ResponseBreakdownInner,
)

from neuronpedia_autointerp.utils import (
    calculate_balanced_accuracy,
    convert_classifier_output_to_score_classifier_output,
    convert_embedding_output_to_score_embedding_output,
    per_feature_scores_embedding,
    per_feature_scores_fuzz_detection,
)


def test_per_feature_scores_embedding():
    score_data = [
        {"distance": -1, "similarity": 0.1},
        {"distance": 1, "similarity": 0.9},
        {"distance": -2, "similarity": 0.2},
        {"distance": 2, "similarity": 0.8},
    ]
    auc_score = per_feature_scores_embedding(score_data)
    assert pytest.approx(auc_score) == 1.0


def test_calculate_balanced_accuracy_positives_and_negatives():
    df = pd.DataFrame(
        [
            {"activating": True, "correct": True},  # TP
            {"activating": True, "correct": False},  # FN
            {"activating": False, "correct": True},  # TN
            {"activating": False, "correct": False},  # FP
        ]
    )
    balanced_accuracy = calculate_balanced_accuracy(df)
    # recall = 1/2
    # specificity = 1/2
    # balanced accuracy = (0.5 + 0.5)/2 = 0.5
    assert pytest.approx(balanced_accuracy) == 0.5


def test_calculate_balanced_accuracy_no_positives():
    df = pd.DataFrame(
        [
            {"activating": False, "correct": True},  # TN
            {"activating": False, "correct": False},  # FP
        ]
    )
    balanced_accuracy = calculate_balanced_accuracy(df)
    # recall = 0
    # specificity = 1/2
    # balanced accuracy = (0 + 0.5)/2 = 0.25
    assert balanced_accuracy == 0.25


def test_calculate_balanced_accuracy_no_negatives():
    df = pd.DataFrame(
        [
            {"activating": True, "correct": True},  # TP
            {"activating": True, "correct": False},  # FP
        ]
    )
    balanced_accuracy = calculate_balanced_accuracy(df)
    assert balanced_accuracy == 0


def test_per_feature_scores_fuzz_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch pd.DataFrame to convert Pydantic models to dicts if needed
    original_dataframe = pd.DataFrame

    def dataframe_wrapper(data: list[Any], *args: Any, **kwargs: Any) -> pd.DataFrame:
        if data and hasattr(data[0], "dict"):
            data = [d.model_dump() for d in data]
        return original_dataframe(data, *args, **kwargs)

    monkeypatch.setattr(pd, "DataFrame", dataframe_wrapper)

    response_1 = ScoreFuzzDetectionPost200ResponseBreakdownInner(
        activating=True,
        prediction=True,
        correct=True,
    )
    response_2 = ScoreFuzzDetectionPost200ResponseBreakdownInner(
        activating=False,
        prediction=False,
        correct=True,
    )
    # We use model_construct for invalid_obj to bypass Pydantic's validation,
    # since the 'prediction' field is expected to be a boolean. By passing -1,
    # which is invalid according to the model, we can test how our function handles
    # this error state
    response_3 = ScoreFuzzDetectionPost200ResponseBreakdownInner.model_construct(
        activating=False,
        prediction=-1,
        correct=False,
    )

    score_data = [response_1, response_2, response_3]
    balanced_accuracy = per_feature_scores_fuzz_detection(score_data)
    assert pytest.approx(balanced_accuracy) == 1.0


def test_convert_classifier_output_no_prediction():
    expected_response = ScoreFuzzDetectionPost200ResponseBreakdownInner.model_construct(
        str_tokens=["test"],
        activations=[0.1],
        distance=1,
        activating=True,
        prediction=None,
        probability=0.7,
        correct=False,
    )
    response = convert_classifier_output_to_score_classifier_output(expected_response)
    assert response.prediction is False
    assert response.str_tokens == expected_response.str_tokens
    assert response.activations == expected_response.activations
    assert response.distance == expected_response.distance
    assert response.activating == expected_response.activating
    assert response.probability == expected_response.probability
    assert response.correct == expected_response.correct


def test_convert_classifier_output_true_prediction():
    expected_response = ScoreFuzzDetectionPost200ResponseBreakdownInner(
        str_tokens=["test2"],
        activations=[0.2],
        distance=2,
        activating=False,
        prediction=True,
        probability=0.8,
        correct=True,
    )
    response = convert_classifier_output_to_score_classifier_output(expected_response)
    assert response.prediction is True


def test_convert_embedding_output_to_score_embedding_output():
    expected_response = ScoreEmbeddingPost200ResponseBreakdownInner(
        text="sample text",
        distance=1,
        similarity=0.6,
    )
    response = convert_embedding_output_to_score_embedding_output(expected_response)
    assert response.text == expected_response.text
    assert response.distance == expected_response.distance
    assert response.similarity == expected_response.similarity
