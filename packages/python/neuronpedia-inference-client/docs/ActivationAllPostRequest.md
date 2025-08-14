# ActivationAllPostRequest

For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompt** | **str** | Input text prompt to get activations for | 
**model** | **str** | Name of the model to test activations on | 
**source_set** | **str** | The source set name of the SAEs (eg gemmascope-res-16k) | 
**selected_sources** | **List[str]** | List of specific SAEs to get activations for (eg [\&quot;0-gemmascope-res-65k\&quot;, \&quot;5-gemmascope-res-65k\&quot;]). If not specified, will get activations for all SAEs in the source set. | [default to []]
**sort_by_token_indexes** | **List[int]** | Sort the results by the sum of the activations at the specified token indexes. | [default to []]
**ignore_bos** | **bool** | Whether or not to include features whose highest activation value is the BOS token. | [default to True]
**feature_filter** | **List[int]** | Optional. If specified, will only return features that match the indexes specified. Can only be used if we&#39;re testing just one SAE (\&quot;selected_sources\&quot; length &#x3D; 1). | [optional] 
**num_results** | **int** | Optional. The number of top features to return. | [optional] [default to 25]
**offset** | **int** | Skip this many top-ranked results before returning the next page. | [optional] [default to 0]

## Example

```python
from neuronpedia_inference_client.models.activation_all_post_request import ActivationAllPostRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationAllPostRequest from a JSON string
activation_all_post_request_instance = ActivationAllPostRequest.from_json(json)
# print the JSON string representation of the object
print(ActivationAllPostRequest.to_json())

# convert the object into a dict
activation_all_post_request_dict = activation_all_post_request_instance.to_dict()
# create an instance of ActivationAllPostRequest from a dict
activation_all_post_request_from_dict = ActivationAllPostRequest.from_dict(activation_all_post_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


