# ActivationSinglePostRequest

Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompt** | **str** | Input text prompt to get activations for | 
**model** | **str** | Name of the model to test activations on | 
**source** | **str** | Source identifier - could be an SAE ID (eg 5-gemmascope-res-16k). Must be specified with \&quot;index\&quot;, or not at all. | [optional] 
**index** | **int** | Index of the SAE. Must be specified with \&quot;source\&quot;, or not at all. | [optional] 
**vector** | **List[float]** | Custom vector to test activations. Must be specified with \&quot;hook\&quot;. | [optional] 
**hook** | **str** | Hook that the custom vector applies to. Must be specified with \&quot;vector\&quot;. | [optional] 

## Example

```python
from neuronpedia_inference_client.models.activation_single_post_request import ActivationSinglePostRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationSinglePostRequest from a JSON string
activation_single_post_request_instance = ActivationSinglePostRequest.from_json(json)
# print the JSON string representation of the object
print(ActivationSinglePostRequest.to_json())

# convert the object into a dict
activation_single_post_request_dict = activation_single_post_request_instance.to_dict()
# create an instance of ActivationSinglePostRequest from a dict
activation_single_post_request_from_dict = ActivationSinglePostRequest.from_dict(activation_single_post_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


