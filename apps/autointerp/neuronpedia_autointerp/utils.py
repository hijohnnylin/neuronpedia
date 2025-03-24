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
    data_df["ground_truth"] = data_df["distance"] > 0
    auc_score = roc_auc_score(data_df["ground_truth"], data_df["similarity"])
    return float(auc_score)  # noqa: RET504


def calculate_balanced_accuracy(dataframe: pd.DataFrame) -> float:
    # ruff: noqa: E712
    # flake8: noqa: E712
    # TODO: "== True" checks should ideally be "is True" but when we check that way, it crashes
    tp = len(
        dataframe[(dataframe["ground_truth"] == True) & (dataframe["correct"] == True)]
    )
    tn = len(
        dataframe[(dataframe["ground_truth"] == False) & (dataframe["correct"] == True)]
    )
    fp = len(
        dataframe[
            (dataframe["ground_truth"] == False) & (dataframe["correct"] == False)
        ]
    )
    fn = len(
        dataframe[(dataframe["ground_truth"] == True) & (dataframe["correct"] == False)]
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
    # if prediction is -1, count it as false (it's an error state)
    # https://github.com/EleutherAI/sae-auto-interp/issues/46
    # TODO: fix this in sae-auto-interp - it should be a boolean as specified in: https://github.com/EleutherAI/sae-auto-interp/blob/3659ff3bfefbe2628d37484e5bcc0087a5b10a27/sae_auto_interp/scorers/classifier/sample.py#L19
    if classifier_output.prediction == -1:
        classifier_output.prediction = False
    return ScoreFuzzDetectionPost200ResponseBreakdownInner(
        str_tokens=classifier_output.str_tokens,
        activations=classifier_output.activations,
        distance=classifier_output.distance,
        ground_truth=classifier_output.ground_truth,
        prediction=bool(classifier_output.prediction),
        highlighted=classifier_output.highlighted,
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
