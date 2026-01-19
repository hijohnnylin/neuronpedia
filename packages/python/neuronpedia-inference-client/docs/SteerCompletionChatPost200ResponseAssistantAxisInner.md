# SteerCompletionChatPost200ResponseAssistantAxisInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | [**NPSteerType**](NPSteerType.md) |  | [optional] 
**pc_titles** | **List[str]** | List of principal component titles/descriptions | [optional] 
**turns** | [**List[SteerCompletionChatPost200ResponseAssistantAxisInnerTurnsInner]**](SteerCompletionChatPost200ResponseAssistantAxisInnerTurnsInner.md) |  | [optional] 

## Example

```python
from neuronpedia_inference_client.models.steer_completion_chat_post200_response_assistant_axis_inner import SteerCompletionChatPost200ResponseAssistantAxisInner

# TODO update the JSON string below
json = "{}"
# create an instance of SteerCompletionChatPost200ResponseAssistantAxisInner from a JSON string
steer_completion_chat_post200_response_assistant_axis_inner_instance = SteerCompletionChatPost200ResponseAssistantAxisInner.from_json(json)
# print the JSON string representation of the object
print(SteerCompletionChatPost200ResponseAssistantAxisInner.to_json())

# convert the object into a dict
steer_completion_chat_post200_response_assistant_axis_inner_dict = steer_completion_chat_post200_response_assistant_axis_inner_instance.to_dict()
# create an instance of SteerCompletionChatPost200ResponseAssistantAxisInner from a dict
steer_completion_chat_post200_response_assistant_axis_inner_from_dict = SteerCompletionChatPost200ResponseAssistantAxisInner.from_dict(steer_completion_chat_post200_response_assistant_axis_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


