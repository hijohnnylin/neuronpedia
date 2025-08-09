# NPLogprobTop

A single top logprob candidate

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**token** | **str** | The token string | 
**logprob** | **float** | The log probability of this token | 

## Example

```python
from neuronpedia_inference_client.models.np_logprob_top import NPLogprobTop

# TODO update the JSON string below
json = "{}"
# create an instance of NPLogprobTop from a JSON string
np_logprob_top_instance = NPLogprobTop.from_json(json)
# print the JSON string representation of the object
print(NPLogprobTop.to_json())

# convert the object into a dict
np_logprob_top_dict = np_logprob_top_instance.to_dict()
# create an instance of NPLogprobTop from a dict
np_logprob_top_from_dict = NPLogprobTop.from_dict(np_logprob_top_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


