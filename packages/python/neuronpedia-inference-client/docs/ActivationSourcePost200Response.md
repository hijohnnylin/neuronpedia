# ActivationSourcePost200Response

All prompts results, only including non-zero features and non-zero activations

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[ActivationSourcePost200ResponseResultsInner]**](ActivationSourcePost200ResponseResultsInner.md) |  | 

## Example

```python
from neuronpedia_inference_client.models.activation_source_post200_response import ActivationSourcePost200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationSourcePost200Response from a JSON string
activation_source_post200_response_instance = ActivationSourcePost200Response.from_json(json)
# print the JSON string representation of the object
print(ActivationSourcePost200Response.to_json())

# convert the object into a dict
activation_source_post200_response_dict = activation_source_post200_response_instance.to_dict()
# create an instance of ActivationSourcePost200Response from a dict
activation_source_post200_response_from_dict = ActivationSourcePost200Response.from_dict(activation_source_post200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


