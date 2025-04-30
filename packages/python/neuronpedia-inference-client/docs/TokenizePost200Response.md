# TokenizePost200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tokens** | **List[int]** | Token IDs for the input text | 
**token_strings** | **List[str]** | String representation of each token | 
**prepend_bos** | **bool** | Whether beginning-of-sequence token was prepended | 

## Example

```python
from neuronpedia_inference_client.models.tokenize_post200_response import TokenizePost200Response

# TODO update the JSON string below
json = "{}"
# create an instance of TokenizePost200Response from a JSON string
tokenize_post200_response_instance = TokenizePost200Response.from_json(json)
# print the JSON string representation of the object
print(TokenizePost200Response.to_json())

# convert the object into a dict
tokenize_post200_response_dict = tokenize_post200_response_instance.to_dict()
# create an instance of TokenizePost200Response from a dict
tokenize_post200_response_from_dict = TokenizePost200Response.from_dict(tokenize_post200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


