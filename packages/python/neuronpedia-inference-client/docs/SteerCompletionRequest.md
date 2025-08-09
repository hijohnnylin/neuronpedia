# SteerCompletionRequest

Base request for steering

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompt** | **str** | Text to pass the model for completion | 
**model** | **str** | Name of the model | 
**steer_method** | [**NPSteerMethod**](NPSteerMethod.md) |  | 
**normalize_steering** | **bool** |  | [default to False]
**types** | [**List[NPSteerType]**](NPSteerType.md) | Array that specifies whether or not to generate STEERED output, DEFAULT (non-steered) output, or both. | 
**features** | [**List[NPSteerFeature]**](NPSteerFeature.md) | Features to steer towards or away from | [optional] 
**vectors** | [**List[NPSteerVector]**](NPSteerVector.md) |  | [optional] 
**n_completion_tokens** | **int** | Number of completion tokens to generate | 
**temperature** | **float** |  | 
**strength_multiplier** | **float** | The steering strength will be multiplied by this number | 
**freq_penalty** | **float** |  | 
**seed** | **float** |  | 
**stream** | **bool** | Whether or not to stream responses using Server Side Events (SSE). Note that the OpenAPI spec does not support SSE - you will receive multiple responses with the same format as non-streaming, except with the \&quot;output\&quot; field chunked. | [optional] [default to False]
**n_logprobs** | **int** | Number of logprobs to return per token. 0 means no logprobs. | [optional] [default to 0]

## Example

```python
from neuronpedia_inference_client.models.steer_completion_request import SteerCompletionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SteerCompletionRequest from a JSON string
steer_completion_request_instance = SteerCompletionRequest.from_json(json)
# print the JSON string representation of the object
print(SteerCompletionRequest.to_json())

# convert the object into a dict
steer_completion_request_dict = steer_completion_request_instance.to_dict()
# create an instance of SteerCompletionRequest from a dict
steer_completion_request_from_dict = SteerCompletionRequest.from_dict(steer_completion_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


