import traceback

import torch
from fastapi import HTTPException
from neuronpedia_autointerp_client.models.score_embedding_post200_response import (
    ScoreEmbeddingPost200Response,
)
from neuronpedia_autointerp_client.models.score_embedding_post_request import (
    ScoreEmbeddingPostRequest,
)
from sae_auto_interp.features import Example, Feature, FeatureRecord
from sae_auto_interp.scorers import EmbeddingScorer
from sae_auto_interp.scorers.scorer import ScorerResult

from neuronpedia_autointerp.utils import (
    convert_embedding_output_to_score_embedding_output,
    per_feature_scores_embedding,
)


async def generate_score_embedding(request: ScoreEmbeddingPostRequest, model):  # type: ignore
    """
    Generate a score for a given set of activations and explanation. This endpoint expects:

    Parameters:
    - activations: A list of dictionaries, each containing a 'tokens' key with a list of token strings
                  and a 'values' key with a list of activation values.
    - explanation: The explanation to use for the score.
    - secret: The secret to authenticate the request.

    Returns a score based on embedding similarity and a detailed breakdown of the scoring.
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

        scorer = EmbeddingScorer(model)
        result: ScorerResult = await scorer.__call__(feature_record)
        score = per_feature_scores_embedding(result.score)
        breakdown = [
            convert_embedding_output_to_score_embedding_output(item)
            for item in result.score
        ]

        return ScoreEmbeddingPost200Response(score=score, breakdown=breakdown)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
