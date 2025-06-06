import traceback

import torch
from delphi.latents.latents import (
    Example,
    Latent,
    LatentRecord,
)
from delphi.scorers import EmbeddingScorer
from delphi.scorers.scorer import ScorerResult
from fastapi import HTTPException
from neuronpedia_autointerp_client.models.score_embedding_post200_response import (
    ScoreEmbeddingPost200Response,
)
from neuronpedia_autointerp_client.models.score_embedding_post_request import (
    ScoreEmbeddingPostRequest,
)

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
        feature = Latent("feature", 0)
        activating_examples = []
        non_activating_examples = []

        for activation in request.activations:
            example = Example(
                tokens=activation.tokens,  # type: ignore
                activations=torch.tensor(activation.values),
            )
            if sum(activation.values) > 0:
                activating_examples.append(
                    [
                        example
                    ]  # TODO: remove brackets once https://github.com/EleutherAI/delphi/issues/132 is fixed
                )
            else:
                non_activating_examples.append(example)

        feature_record = LatentRecord(feature)
        feature_record.test = activating_examples
        feature_record.not_active = non_activating_examples
        feature_record.extra_examples = non_activating_examples
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
