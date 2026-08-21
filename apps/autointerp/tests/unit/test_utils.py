"""Tests for the scorer-output aggregation helpers.

These take sae-auto-interp's own dataclasses, which is what ``ScorerResult.score``
actually holds, so the fixtures below build those rather than our ``schemas`` models.
Feeding in the wire models instead would silently change what pandas does with them --
it reads field names off a dataclass but not off a pydantic model -- so using the real
type is what keeps these tests honest about the production path.
"""

import pandas as pd
import pytest
from sae_auto_interp.scorers.classifier.sample import ClassifierOutput
from sae_auto_interp.scorers.embedding.embedding import EmbeddingOutput

from neuronpedia_autointerp.utils import (
    ScoringInputError,
    calculate_balanced_accuracy,
    convert_classifier_output_to_score_classifier_output,
    convert_embedding_output_to_score_embedding_output,
    per_feature_scores_embedding,
    per_feature_scores_fuzz_detection,
)


def classifier_output(
    *,
    ground_truth: bool,
    correct: bool,
    prediction: bool | int = True,
    str_tokens: list[str] | None = None,
    activations: list[float] | None = None,
    distance: int = 1,
    highlighted: bool = False,
    probability: float = 0.5,
) -> ClassifierOutput:
    """One scored example, with everything the tests don't care about defaulted."""
    return ClassifierOutput(
        str_tokens=str_tokens if str_tokens is not None else ["tok"],
        activations=activations if activations is not None else [0.0],
        distance=distance,
        ground_truth=ground_truth,
        # Declared bool, but sae-auto-interp writes -1 here when the scorer fails to
        # answer, which is the case the callers below need to be able to construct.
        prediction=prediction,  # pyright: ignore[reportArgumentType]
        highlighted=highlighted,
        probability=probability,
        correct=correct,
    )


def accuracy_frame(rows: list[tuple[bool, bool]]) -> pd.DataFrame:
    """Build the (ground_truth, correct) frame that balanced accuracy reads."""
    return pd.DataFrame([{"ground_truth": truth, "correct": correct} for truth, correct in rows])


class TestCalculateBalancedAccuracy:
    def test_averages_recall_and_specificity(self):
        # One of each quadrant: recall 1/2, specificity 1/2, so the mean is 1/2.
        score = calculate_balanced_accuracy(
            accuracy_frame([(True, True), (True, False), (False, True), (False, False)])
        )
        assert score == pytest.approx(0.5)

    def test_perfect_predictions_score_one(self):
        score = calculate_balanced_accuracy(accuracy_frame([(True, True), (False, True)]))
        assert score == pytest.approx(1.0)

    def test_everything_wrong_scores_zero(self):
        score = calculate_balanced_accuracy(accuracy_frame([(True, False), (False, False)]))
        assert score == pytest.approx(0.0)

    def test_without_positive_examples_recall_counts_as_zero(self):
        # No positives means recall is undefined; it is treated as 0 rather than
        # dropped, so a perfect score on the negatives still only reaches half.
        score = calculate_balanced_accuracy(accuracy_frame([(False, True), (False, False)]))
        assert score == pytest.approx(0.25)

    def test_without_negative_examples_result_is_zero(self):
        # Specificity is undefined with no negatives, and the guard short-circuits the
        # whole result to 0 -- note this discards the recall term, unlike the case above.
        score = calculate_balanced_accuracy(accuracy_frame([(True, True), (True, False)]))
        assert score == 0


class TestPerFeatureScoresFuzzDetection:
    def test_scores_dataclass_rows_without_help(self):
        # Guards the pandas/dataclass coupling: if these stopped being dataclasses,
        # the columns would come back as integers and this would raise KeyError.
        score = per_feature_scores_fuzz_detection(
            [
                classifier_output(ground_truth=True, correct=True),
                classifier_output(ground_truth=False, correct=True),
            ]
        )
        assert score == pytest.approx(1.0)

    def test_error_rows_are_excluded_before_scoring(self):
        # prediction == -1 is sae-auto-interp's error marker, not a real prediction.
        # Dropping it leaves a clean split, so a row that would otherwise drag the
        # score down has no effect at all.
        rows = [
            classifier_output(ground_truth=True, correct=True),
            classifier_output(ground_truth=False, correct=True),
            classifier_output(ground_truth=False, correct=False, prediction=-1),
        ]
        assert per_feature_scores_fuzz_detection(rows) == pytest.approx(1.0)

    def test_every_row_erroring_is_rejected(self):
        # Dropping the -1 rows can leave nothing at all, which used to reach pandas as an
        # empty frame and raise KeyError('ground_truth') from inside the accuracy helper.
        with pytest.raises(ScoringInputError):
            per_feature_scores_fuzz_detection(
                [
                    classifier_output(ground_truth=True, correct=False, prediction=-1),
                    classifier_output(ground_truth=False, correct=False, prediction=-1),
                ]
            )

    def test_false_predictions_are_kept(self):
        # A False prediction must not be mistaken for the -1 error marker.
        rows = [
            classifier_output(ground_truth=True, correct=False, prediction=False),
            classifier_output(ground_truth=False, correct=True, prediction=False),
        ]
        assert per_feature_scores_fuzz_detection(rows) == pytest.approx(0.5)


class TestPerFeatureScoresEmbedding:
    def test_ranks_similar_examples_above_distant_ones(self):
        # distance > 0 is the positive class; similarity is the score being ranked.
        # Here every positive outranks every negative, which is a perfect AUC.
        score = per_feature_scores_embedding(
            [
                EmbeddingOutput(text="near", distance=1, similarity=0.9),
                EmbeddingOutput(text="near", distance=2, similarity=0.8),
                EmbeddingOutput(text="far", distance=-1, similarity=0.1),
                EmbeddingOutput(text="far", distance=-2, similarity=0.2),
            ]
        )
        assert score == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        score = per_feature_scores_embedding(
            [
                EmbeddingOutput(text="near", distance=1, similarity=0.1),
                EmbeddingOutput(text="far", distance=-1, similarity=0.9),
            ]
        )
        assert score == pytest.approx(0.0)

    def test_single_class_is_rejected_rather_than_scored(self):
        # AUC is undefined when every example is the same class, and sklearn answers nan
        # rather than raising. Letting that through means a nan score, which json.dumps
        # refuses to encode -- so the request would fail at serialization time with
        # "Out of range float values are not JSON compliant" instead of here.
        with pytest.raises(ScoringInputError):
            per_feature_scores_embedding(
                [
                    EmbeddingOutput(text="near", distance=1, similarity=0.9),
                    EmbeddingOutput(text="near", distance=2, similarity=0.8),
                ]
            )

    def test_only_negatives_is_rejected_too(self):
        # The mirror image: nothing activating to rank the non-activating text against.
        with pytest.raises(ScoringInputError):
            per_feature_scores_embedding([EmbeddingOutput(text="far", distance=-1, similarity=0.1)])

    def test_no_examples_at_all_is_rejected(self):
        with pytest.raises(ScoringInputError):
            per_feature_scores_embedding([])


class TestConvertClassifierOutput:
    def test_error_prediction_becomes_false(self):
        # -1 means the scorer failed to answer; it is recorded as a negative rather
        # than leaking a non-boolean through to the response model.
        converted = convert_classifier_output_to_score_classifier_output(
            classifier_output(ground_truth=True, correct=False, prediction=-1)
        )
        assert converted.prediction is False

    def test_carries_every_field_across(self):
        source = classifier_output(
            ground_truth=False,
            correct=True,
            prediction=True,
            str_tokens=["hello", "world"],
            activations=[0.25, 0.75],
            distance=3,
            highlighted=True,
            probability=0.8,
        )
        converted = convert_classifier_output_to_score_classifier_output(source)
        assert converted.prediction is True
        assert converted.str_tokens == source.str_tokens
        assert converted.activations == source.activations
        assert converted.distance == source.distance
        assert converted.ground_truth == source.ground_truth
        assert converted.highlighted == source.highlighted
        assert converted.probability == source.probability
        assert converted.correct == source.correct


class TestConvertEmbeddingOutput:
    def test_carries_every_field_across(self):
        source = EmbeddingOutput(text="sample text", distance=1, similarity=0.6)
        converted = convert_embedding_output_to_score_embedding_output(source)
        assert converted.text == source.text
        assert converted.distance == source.distance
        assert converted.similarity == source.similarity
