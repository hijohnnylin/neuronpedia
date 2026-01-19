# SteerCompletionChatPost200Response

The steering/default chat responses.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**assistant_axis** | [**List[SteerCompletionChatPost200ResponseAssistantAxisInner]**](SteerCompletionChatPost200ResponseAssistantAxisInner.md) | Persona monitoring data for assistant turns, one entry per steer type | [optional] 
**outputs** | [**List[NPSteerChatResult]**](NPSteerChatResult.md) |  | 
**input** | [**NPSteerChatResult**](NPSteerChatResult.md) |  | 

## Example

```python
from neuronpedia_inference_client.models.steer_completion_chat_post200_response import SteerCompletionChatPost200Response

# TODO update the JSON string below
json = "{}"
# create an instance of SteerCompletionChatPost200Response from a JSON string
steer_completion_chat_post200_response_instance = SteerCompletionChatPost200Response.from_json(json)
# print the JSON string representation of the object
print(SteerCompletionChatPost200Response.to_json())

# convert the object into a dict
steer_completion_chat_post200_response_dict = steer_completion_chat_post200_response_instance.to_dict()
# create an instance of SteerCompletionChatPost200Response from a dict
steer_completion_chat_post200_response_from_dict = SteerCompletionChatPost200Response.from_dict(steer_completion_chat_post200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


