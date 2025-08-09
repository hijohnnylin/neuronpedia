# NPSteerChatResult

The formatted and unformatted (\"raw\") chat messages

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**chat_template** | [**List[NPSteerChatMessage]**](NPSteerChatMessage.md) |  | 
**raw** | **str** |  | 
**type** | [**NPSteerType**](NPSteerType.md) |  | [optional] 
**logprobs** | [**List[NPLogprob]**](NPLogprob.md) | Token logprobs for the output sequence. Only present if n_logprobs &gt; 0. | [optional] 

## Example

```python
from neuronpedia_inference_client.models.np_steer_chat_result import NPSteerChatResult

# TODO update the JSON string below
json = "{}"
# create an instance of NPSteerChatResult from a JSON string
np_steer_chat_result_instance = NPSteerChatResult.from_json(json)
# print the JSON string representation of the object
print(NPSteerChatResult.to_json())

# convert the object into a dict
np_steer_chat_result_dict = np_steer_chat_result_instance.to_dict()
# create an instance of NPSteerChatResult from a dict
np_steer_chat_result_from_dict = NPSteerChatResult.from_dict(np_steer_chat_result_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


