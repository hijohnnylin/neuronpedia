import traceback

import torch
from fastapi import HTTPException
from neuronpedia_autointerp_client.models.explain_default_post200_response import (
    ExplainDefaultPost200Response,
)
from neuronpedia_autointerp_client.models.explain_default_post_request import (
    ExplainDefaultPostRequest,
)
from sae_auto_interp.clients import OpenRouter
from sae_auto_interp.explainers import DefaultExplainer
from sae_auto_interp.explainers.explainer import ExplainerResult
from sae_auto_interp.features import Example, Feature, FeatureRecord


async def explain_default(request: ExplainDefaultPostRequest):
    """
    Generate an explanation for a given set of activations.
    """
    try:
        feature = Feature("feature", 0)
        examples = []
        for activation in request.activations:
            example = Example(activation.tokens, torch.tensor(activation.values))  # type: ignore
            examples.append(example)
        feature_record = FeatureRecord(feature)
        feature_record.train = examples

        client = OpenRouter(api_key=request.openrouter_key, model=request.model)
        explainer = DefaultExplainer(client, tokenizer=None, threshold=0.6)
        result: ExplainerResult = await explainer.__call__(feature_record)  # type: ignore

        return ExplainDefaultPost200Response(explanation=result.explanation)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
