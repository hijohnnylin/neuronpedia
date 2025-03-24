import traceback

import torch
from fastapi import HTTPException
from neuronpedia_autointerp.utils import (
    convert_classifier_output_to_score_classifier_output,
    per_feature_scores_fuzz_detection,
)
from neuronpedia_autointerp_client.models.np_score_fuzz_detection_type import (
    NPScoreFuzzDetectionType,
)
from neuronpedia_autointerp_client.models.score_fuzz_detection_post200_response import (
    ScoreFuzzDetectionPost200Response,
)
from neuronpedia_autointerp_client.models.score_fuzz_detection_post_request import (
    ScoreFuzzDetectionPostRequest,
)
from sae_auto_interp.clients import OpenRouter
from sae_auto_interp.features import Example, Feature, FeatureRecord
from sae_auto_interp.scorers import DetectionScorer, FuzzingScorer
from sae_auto_interp.scorers.scorer import ScorerResult


async def generate_score_fuzz_detection(request: ScoreFuzzDetectionPostRequest):
    """
    Generate a score for a given set of activations and explanation. This endpoint expects:

    Parameters:
    - activations: A list of dictionaries, each containing a 'tokens' key with a list of token strings
                  and a 'values' key with a list of activation values.
    - explanation: The explanation to use for the score.
    - type: The scoring type to use - either "fuzz" or "detection".
    - openrouter_key: The API key to use for OpenRouter.
    - model: The model to use for the request.
    - secret: Authentication secret.

    Note: OpenRouter doesn't support log_prob, so we can't use that feature.
    We currently show 5 examples at a time (batch_size=5).
    """
    try:
        feature = Feature("feature", 0)
        activating_examples = []
        non_activating_examples = []

        for activation in request.activations:
            example = Example(activation.tokens, torch.tensor(activation.values))  # type: ignore
            if sum(activation.values) > 0:
                activating_examples.append(example)
            else:
                non_activating_examples.append(example)

        feature_record = FeatureRecord(feature)
        feature_record.test = [activating_examples]
        feature_record.extra_examples = non_activating_examples  # type: ignore
        feature_record.random_examples = non_activating_examples  # type: ignore
        feature_record.explanation = request.explanation  # type: ignore

        client = OpenRouter(api_key=request.openrouter_key, model=request.model)

        if request.type == NPScoreFuzzDetectionType.FUZZ:
            scorer = FuzzingScorer(
                client,
                tokenizer=None,  # type: ignore
                batch_size=5,
                verbose=False,
                log_prob=False,
            )
        elif request.type == NPScoreFuzzDetectionType.DETECTION:
            scorer = DetectionScorer(
                client,
                tokenizer=None,  # type: ignore
                batch_size=5,
                verbose=False,
                log_prob=False,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid scoring type")

        result: ScorerResult = await scorer.__call__(feature_record)  # type: ignore
        score = per_feature_scores_fuzz_detection(result.score)

        breakdown = [
            convert_classifier_output_to_score_classifier_output(item)
            for item in result.score
        ]

        return ScoreFuzzDetectionPost200Response(score=score, breakdown=breakdown)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
