# NPLogprob

Logprobs for a single token

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**token** | **str** | The chosen token | 
**logprob** | **float** | The log probability of the chosen token | 
**top_logprobs** | [**List[NPLogprobTop]**](NPLogprobTop.md) | Top candidate tokens and their log probabilities | 

## Example

```python
from neuronpedia_inference_client.models.np_logprob import NPLogprob

# TODO update the JSON string below
json = "{}"
# create an instance of NPLogprob from a JSON string
np_logprob_instance = NPLogprob.from_json(json)
# print the JSON string representation of the object
print(NPLogprob.to_json())

# convert the object into a dict
np_logprob_dict = np_logprob_instance.to_dict()
# create an instance of NPLogprob from a dict
np_logprob_from_dict = NPLogprob.from_dict(np_logprob_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


