# ruff: noqa: E712

"""Turn a scorer's per-example output into a single score plus a wire-safe breakdown.

The inputs here come straight off ``ScorerResult.score``, so they are sae-auto-interp's
own dataclasses rather than our wire models. That distinction matters: the aggregation
below leans on ``pd.DataFrame`` reading field names off a dataclass, which it will not
do for a pydantic model. The convert_* functions are the boundary where those dataclasses
become the ``schemas`` models that define the response format.
"""

import pandas as pd
from sae_auto_interp.scorers.classifier.sample import ClassifierOutput
from sae_auto_interp.scorers.embedding.embedding import EmbeddingOutput
from sklearn.metrics import roc_auc_score

from neuronpedia_autointerp.schemas import (
    ScoreEmbeddingBreakdownItem,
    ScoreFuzzDetectionBreakdownItem,
)


class ScoringInputError(ValueError):
    """The request parsed, but its examples leave the metric undefined.

    The routes turn this into a 400 rather than a 500: nothing on the server went wrong,
    the caller sent examples no score can be computed from. Without it these cases reach
    the client as an unhandled ``nan`` or ``KeyError`` from deep inside pandas.
    """


def per_feature_scores_embedding(score_data: list[EmbeddingOutput]) -> float:
    """Score how well similarity to the explanation separates activating from non-activating text."""
    if not score_data:
        raise ScoringInputError("Embedding scoring needs at least one example, and this request had none.")
    data_df = pd.DataFrame(score_data)
    data_df["ground_truth"] = data_df["distance"] > 0
    # roc_auc_score answers nan rather than raising when y_true holds a single class, and nan is
    # not JSON, so the response would fail to encode at the very end of the request instead.
    if data_df["ground_truth"].nunique() < 2:
        raise ScoringInputError(
            "Embedding scoring needs both activating and non-activating examples to compare, "
            "and this request had only one kind."
        )
    auc_score = float(roc_auc_score(data_df["ground_truth"], data_df["similarity"]))
    return auc_score  # noqa: RET504


def calculate_balanced_accuracy(dataframe: pd.DataFrame) -> float:
    tp = len(dataframe[(dataframe["ground_truth"] == True) & (dataframe["correct"] == True)])
    tn = len(dataframe[(dataframe["ground_truth"] == False) & (dataframe["correct"] == True)])
    fp = len(dataframe[(dataframe["ground_truth"] == False) & (dataframe["correct"] == False)])
    fn = len(dataframe[(dataframe["ground_truth"] == True) & (dataframe["correct"] == False)])
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    return 0 if tn + fp == 0 else (recall + tn / (tn + fp)) / 2


def per_feature_scores_fuzz_detection(
    score_data: list[ClassifierOutput],
) -> float:
    """Score the fuzz/detection predictions, ignoring the examples the scorer failed to answer."""
    data = [d for d in score_data if d.prediction != -1]
    if not data:
        # Every example carries sae-auto-interp's -1 error marker, so the scorer model answered
        # nothing usable at all -- a rejected API key or an exhausted account looks like this.
        raise ScoringInputError(
            "The scorer model returned no usable predictions. Check that the OpenRouter key is valid and has credits."
        )
    data_df = pd.DataFrame(data)
    balanced_accuracy = calculate_balanced_accuracy(data_df)
    return balanced_accuracy  # noqa: RET504


def convert_classifier_output_to_score_classifier_output(
    classifier_output: ClassifierOutput,
) -> ScoreFuzzDetectionBreakdownItem:
    # if prediction is -1, count it as false (it's an error state)
    # https://github.com/EleutherAI/sae-auto-interp/issues/46
    # TODO: fix this in sae-auto-interp - it should be a boolean as specified in: https://github.com/EleutherAI/sae-auto-interp/blob/3659ff3bfefbe2628d37484e5bcc0087a5b10a27/sae_auto_interp/scorers/classifier/sample.py#L19
    if classifier_output.prediction == -1:
        classifier_output.prediction = False
    return ScoreFuzzDetectionBreakdownItem(
        str_tokens=classifier_output.str_tokens,
        activations=classifier_output.activations,
        # The scorer types distance as float|int while the schema says integer. Every
        # scorer we call sets a quantile index, so this holds in practice; passing it
        # through unconverted means a genuine float fails loudly here rather than being
        # silently truncated into a wrong quantile.
        distance=classifier_output.distance,  # pyright: ignore[reportArgumentType]
        ground_truth=classifier_output.ground_truth,
        prediction=bool(classifier_output.prediction),
        highlighted=classifier_output.highlighted,
        probability=classifier_output.probability,
        correct=classifier_output.correct,
    )


def convert_embedding_output_to_score_embedding_output(
    embedding_output: EmbeddingOutput,
) -> ScoreEmbeddingBreakdownItem:
    return ScoreEmbeddingBreakdownItem(
        text=embedding_output.text,
        # Same float|int-vs-integer mismatch as the classifier conversion above.
        distance=embedding_output.distance,  # pyright: ignore[reportArgumentType]
        similarity=embedding_output.similarity,
    )
