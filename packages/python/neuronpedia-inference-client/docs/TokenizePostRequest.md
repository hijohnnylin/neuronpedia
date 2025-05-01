# TokenizePostRequest

Tokenize input text for a given model

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**model** | **str** | ID of the model to use for tokenization | 
**text** | **str** | The text to tokenize | 
**prepend_bos** | **bool** | Whether to prepend beginning-of-sequence token. If not specified, uses the model&#39;s default setting. | [optional] 

## Example

```python
from neuronpedia_inference_client.models.tokenize_post_request import TokenizePostRequest

# TODO update the JSON string below
json = "{}"
# create an instance of TokenizePostRequest from a JSON string
tokenize_post_request_instance = TokenizePostRequest.from_json(json)
# print the JSON string representation of the object
print(TokenizePostRequest.to_json())

# convert the object into a dict
tokenize_post_request_dict = tokenize_post_request_instance.to_dict()
# create an instance of TokenizePostRequest from a dict
tokenize_post_request_from_dict = TokenizePostRequest.from_dict(tokenize_post_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


