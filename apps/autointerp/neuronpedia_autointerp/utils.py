# ruff: noqa: E712

from typing import Any

import pandas as pd
from neuronpedia_autointerp_client.models.score_embedding_post200_response_breakdown_inner import (
    ScoreEmbeddingPost200ResponseBreakdownInner,
)
from neuronpedia_autointerp_client.models.score_fuzz_detection_post200_response_breakdown_inner import (
    ScoreFuzzDetectionPost200ResponseBreakdownInner,
)
from sklearn.metrics import roc_auc_score


def per_feature_scores_embedding(score_data: list[dict[Any, Any]]) -> float:
    data_df = pd.DataFrame(score_data)
    data_df["activating"] = data_df["distance"] > 0
    auc_score = float(roc_auc_score(data_df["activating"], data_df["similarity"]))
    return auc_score  # noqa: RET504


def calculate_balanced_accuracy(dataframe: pd.DataFrame) -> float:
    tp = len(
        dataframe[(dataframe["activating"] == True) & (dataframe["correct"] == True)]
    )
    tn = len(
        dataframe[(dataframe["activating"] == False) & (dataframe["correct"] == True)]
    )
    fp = len(
        dataframe[(dataframe["activating"] == False) & (dataframe["correct"] == False)]
    )
    fn = len(
        dataframe[(dataframe["activating"] == True) & (dataframe["correct"] == False)]
    )
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    return 0 if tn + fp == 0 else (recall + tn / (tn + fp)) / 2


def per_feature_scores_fuzz_detection(
    score_data: list[ScoreFuzzDetectionPost200ResponseBreakdownInner],
) -> float:
    data = [d for d in score_data if d.prediction != -1]
    data_df = pd.DataFrame(data)
    balanced_accuracy = calculate_balanced_accuracy(data_df)
    return balanced_accuracy  # noqa: RET504


def convert_classifier_output_to_score_classifier_output(
    classifier_output: ScoreFuzzDetectionPost200ResponseBreakdownInner,
) -> ScoreFuzzDetectionPost200ResponseBreakdownInner:
    return ScoreFuzzDetectionPost200ResponseBreakdownInner(
        str_tokens=classifier_output.str_tokens,
        activations=classifier_output.activations,
        distance=classifier_output.distance,
        activating=classifier_output.activating,
        prediction=bool(classifier_output.prediction),
        probability=classifier_output.probability,
        correct=classifier_output.correct,
    )


def convert_embedding_output_to_score_embedding_output(
    embedding_output: ScoreEmbeddingPost200ResponseBreakdownInner,
) -> ScoreEmbeddingPost200ResponseBreakdownInner:
    return ScoreEmbeddingPost200ResponseBreakdownInner(
        text=embedding_output.text,
        distance=embedding_output.distance,
        similarity=embedding_output.similarity,
    )
