# ActivationSourcePost200ResponseResultsInner

One prompt's results, only including non-zero values and non-zero activations

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tokens** | **List[str]** | The prompt, tokenized. | 
**active_features** | **Dict[str, List[List[float]]]** | Dictionary mapping feature indices to arrays of [token_index, activation_value] | [optional] 

## Example

```python
from neuronpedia_inference_client.models.activation_source_post200_response_results_inner import ActivationSourcePost200ResponseResultsInner

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationSourcePost200ResponseResultsInner from a JSON string
activation_source_post200_response_results_inner_instance = ActivationSourcePost200ResponseResultsInner.from_json(json)
# print the JSON string representation of the object
print(ActivationSourcePost200ResponseResultsInner.to_json())

# convert the object into a dict
activation_source_post200_response_results_inner_dict = activation_source_post200_response_results_inner_instance.to_dict()
# create an instance of ActivationSourcePost200ResponseResultsInner from a dict
activation_source_post200_response_results_inner_from_dict = ActivationSourcePost200ResponseResultsInner.from_dict(activation_source_post200_response_results_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


