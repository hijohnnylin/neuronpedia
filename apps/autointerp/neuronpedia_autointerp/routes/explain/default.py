import traceback

import torch
from delphi.clients import OpenRouter
from delphi.explainers import DefaultExplainer
from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import Example, Latent, LatentRecord
from fastapi import HTTPException
from neuronpedia_autointerp_client.models.explain_default_post200_response import (
    ExplainDefaultPost200Response,
)
from neuronpedia_autointerp_client.models.explain_default_post_request import (
    ExplainDefaultPostRequest,
)


async def explain_default(request: ExplainDefaultPostRequest):
    """
    Generate an explanation for a given set of activations.
    """
    try:
        feature = Latent("feature", 0)
        examples = []
        for activation in request.activations:
            example = Example(
                tokens=activation.tokens,  # type: ignore
                activations=torch.tensor(activation.values),
                str_tokens=activation.tokens,
            )
            examples.append(example)
        feature_record = LatentRecord(feature)
        feature_record.train = examples

        client = OpenRouter(api_key=request.openrouter_key, model=request.model)
        explainer = DefaultExplainer(client, threshold=0.6, activations=False)
        result: ExplainerResult = await explainer.__call__(feature_record)

        return ExplainDefaultPost200Response(explanation=result.explanation)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
