# NPSteerCompletionResponseInner

A streamed steering/default response. Output is either the whole response or a chunk, depending on response type.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**NPSteerType**](NPSteerType.md) |  | 
**output** | **str** |  | 
**logprobs** | [**List[NPLogprob]**](NPLogprob.md) | Token logprobs for the output sequence. Only present if n_logprobs &gt; 0. | [optional] 

## Example

```python
from neuronpedia_inference_client.models.np_steer_completion_response_inner import NPSteerCompletionResponseInner

# TODO update the JSON string below
json = "{}"
# create an instance of NPSteerCompletionResponseInner from a JSON string
np_steer_completion_response_inner_instance = NPSteerCompletionResponseInner.from_json(json)
# print the JSON string representation of the object
print(NPSteerCompletionResponseInner.to_json())

# convert the object into a dict
np_steer_completion_response_inner_dict = np_steer_completion_response_inner_instance.to_dict()
# create an instance of NPSteerCompletionResponseInner from a dict
np_steer_completion_response_inner_from_dict = NPSteerCompletionResponseInner.from_dict(np_steer_completion_response_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


