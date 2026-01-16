# ActivationSourcePostRequest

For a given prompt, get the top activating features for a source (eg 0-gemmascope-res-65k or 5-gemmascope-res-65k), and return the results as a 3D array of prompt x prompt_token x feature_index.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompts** | **List[str]** | Input text prompt to get activations for | 
**model** | **str** | Name of the model to test activations on | 
**source** | **str** | The source (eg 5-gemmascope-res-16k) | 

## Example

```python
from neuronpedia_inference_client.models.activation_source_post_request import ActivationSourcePostRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationSourcePostRequest from a JSON string
activation_source_post_request_instance = ActivationSourcePostRequest.from_json(json)
# print the JSON string representation of the object
print(ActivationSourcePostRequest.to_json())

# convert the object into a dict
activation_source_post_request_dict = activation_source_post_request_instance.to_dict()
# create an instance of ActivationSourcePostRequest from a dict
activation_source_post_request_from_dict = ActivationSourcePostRequest.from_dict(activation_source_post_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


