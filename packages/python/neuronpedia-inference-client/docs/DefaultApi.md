# neuronpedia_inference_client.DefaultApi

All URIs are relative to */v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**activation_all_post**](DefaultApi.md#activation_all_post) | **POST** /activation/all | For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.
[**activation_single_post**](DefaultApi.md#activation_single_post) | **POST** /activation/single | Given a text prompt, returns the activation values for a single SAE latent or custom vector+hook.
[**activation_topk_by_token_post**](DefaultApi.md#activation_topk_by_token_post) | **POST** /activation/topk-by-token | For a given prompt, get the top activating features at each token position for a single SAE.
[**steer_completion_chat_post**](DefaultApi.md#steer_completion_chat_post) | **POST** /steer/completion-chat | For a given prompt, complete it by steering with the given feature or vector
[**steer_completion_post**](DefaultApi.md#steer_completion_post) | **POST** /steer/completion | For a given prompt, complete it by steering with the given feature or vector
[**tokenize_post**](DefaultApi.md#tokenize_post) | **POST** /tokenize | Tokenize input text for a given model
[**util_sae_topk_by_decoder_cossim_post**](DefaultApi.md#util_sae_topk_by_decoder_cossim_post) | **POST** /util/sae-topk-by-decoder-cossim | Given a specific vector or SAE feature, return the top features by cosine similarity in the same SAE
[**util_sae_vector_post**](DefaultApi.md#util_sae_vector_post) | **POST** /util/sae-vector | Get the raw vector for an SAE feature


# **activation_all_post**
> ActivationAllPost200Response activation_all_post(activation_all_post_request)

For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.activation_all_post200_response import ActivationAllPost200Response
from neuronpedia_inference_client.models.activation_all_post_request import ActivationAllPostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    activation_all_post_request = neuronpedia_inference_client.ActivationAllPostRequest() # ActivationAllPostRequest | 

    try:
        # For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.
        api_response = api_instance.activation_all_post(activation_all_post_request)
        print("The response of DefaultApi->activation_all_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->activation_all_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **activation_all_post_request** | [**ActivationAllPostRequest**](ActivationAllPostRequest.md)|  | 

### Return type

[**ActivationAllPost200Response**](ActivationAllPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved results |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **activation_single_post**
> ActivationSinglePost200Response activation_single_post(activation_single_post_request)

Given a text prompt, returns the activation values for a single SAE latent or custom vector+hook.

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.activation_single_post200_response import ActivationSinglePost200Response
from neuronpedia_inference_client.models.activation_single_post_request import ActivationSinglePostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    activation_single_post_request = neuronpedia_inference_client.ActivationSinglePostRequest() # ActivationSinglePostRequest | 

    try:
        # Given a text prompt, returns the activation values for a single SAE latent or custom vector+hook.
        api_response = api_instance.activation_single_post(activation_single_post_request)
        print("The response of DefaultApi->activation_single_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->activation_single_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **activation_single_post_request** | [**ActivationSinglePostRequest**](ActivationSinglePostRequest.md)|  | 

### Return type

[**ActivationSinglePost200Response**](ActivationSinglePost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved results |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **activation_topk_by_token_post**
> ActivationTopkByTokenPost200Response activation_topk_by_token_post(activation_topk_by_token_post_request)

For a given prompt, get the top activating features at each token position for a single SAE.

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.activation_topk_by_token_post200_response import ActivationTopkByTokenPost200Response
from neuronpedia_inference_client.models.activation_topk_by_token_post_request import ActivationTopkByTokenPostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    activation_topk_by_token_post_request = neuronpedia_inference_client.ActivationTopkByTokenPostRequest() # ActivationTopkByTokenPostRequest | 

    try:
        # For a given prompt, get the top activating features at each token position for a single SAE.
        api_response = api_instance.activation_topk_by_token_post(activation_topk_by_token_post_request)
        print("The response of DefaultApi->activation_topk_by_token_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->activation_topk_by_token_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **activation_topk_by_token_post_request** | [**ActivationTopkByTokenPostRequest**](ActivationTopkByTokenPostRequest.md)|  | 

### Return type

[**ActivationTopkByTokenPost200Response**](ActivationTopkByTokenPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved results |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **steer_completion_chat_post**
> SteerCompletionChatPost200Response steer_completion_chat_post(steer_completion_chat_post_request)

For a given prompt, complete it by steering with the given feature or vector

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.steer_completion_chat_post200_response import SteerCompletionChatPost200Response
from neuronpedia_inference_client.models.steer_completion_chat_post_request import SteerCompletionChatPostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    steer_completion_chat_post_request = neuronpedia_inference_client.SteerCompletionChatPostRequest() # SteerCompletionChatPostRequest | 

    try:
        # For a given prompt, complete it by steering with the given feature or vector
        api_response = api_instance.steer_completion_chat_post(steer_completion_chat_post_request)
        print("The response of DefaultApi->steer_completion_chat_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->steer_completion_chat_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **steer_completion_chat_post_request** | [**SteerCompletionChatPostRequest**](SteerCompletionChatPostRequest.md)|  | 

### Return type

[**SteerCompletionChatPost200Response**](SteerCompletionChatPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved results |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **steer_completion_post**
> SteerCompletionPost200Response steer_completion_post(steer_completion_request)

For a given prompt, complete it by steering with the given feature or vector

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.steer_completion_post200_response import SteerCompletionPost200Response
from neuronpedia_inference_client.models.steer_completion_request import SteerCompletionRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    steer_completion_request = neuronpedia_inference_client.SteerCompletionRequest() # SteerCompletionRequest | 

    try:
        # For a given prompt, complete it by steering with the given feature or vector
        api_response = api_instance.steer_completion_post(steer_completion_request)
        print("The response of DefaultApi->steer_completion_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->steer_completion_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **steer_completion_request** | [**SteerCompletionRequest**](SteerCompletionRequest.md)|  | 

### Return type

[**SteerCompletionPost200Response**](SteerCompletionPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved results |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tokenize_post**
> TokenizePost200Response tokenize_post(tokenize_post_request)

Tokenize input text for a given model

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.tokenize_post200_response import TokenizePost200Response
from neuronpedia_inference_client.models.tokenize_post_request import TokenizePostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    tokenize_post_request = neuronpedia_inference_client.TokenizePostRequest() # TokenizePostRequest | 

    try:
        # Tokenize input text for a given model
        api_response = api_instance.tokenize_post(tokenize_post_request)
        print("The response of DefaultApi->tokenize_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->tokenize_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **tokenize_post_request** | [**TokenizePostRequest**](TokenizePostRequest.md)|  | 

### Return type

[**TokenizePost200Response**](TokenizePost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful tokenization |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **util_sae_topk_by_decoder_cossim_post**
> UtilSaeTopkByDecoderCossimPost200Response util_sae_topk_by_decoder_cossim_post(util_sae_topk_by_decoder_cossim_post_request)

Given a specific vector or SAE feature, return the top features by cosine similarity in the same SAE

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post200_response import UtilSaeTopkByDecoderCossimPost200Response
from neuronpedia_inference_client.models.util_sae_topk_by_decoder_cossim_post_request import UtilSaeTopkByDecoderCossimPostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    util_sae_topk_by_decoder_cossim_post_request = neuronpedia_inference_client.UtilSaeTopkByDecoderCossimPostRequest() # UtilSaeTopkByDecoderCossimPostRequest | 

    try:
        # Given a specific vector or SAE feature, return the top features by cosine similarity in the same SAE
        api_response = api_instance.util_sae_topk_by_decoder_cossim_post(util_sae_topk_by_decoder_cossim_post_request)
        print("The response of DefaultApi->util_sae_topk_by_decoder_cossim_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->util_sae_topk_by_decoder_cossim_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **util_sae_topk_by_decoder_cossim_post_request** | [**UtilSaeTopkByDecoderCossimPostRequest**](UtilSaeTopkByDecoderCossimPostRequest.md)|  | 

### Return type

[**UtilSaeTopkByDecoderCossimPost200Response**](UtilSaeTopkByDecoderCossimPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved results |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **util_sae_vector_post**
> UtilSaeVectorPost200Response util_sae_vector_post(util_sae_vector_post_request)

Get the raw vector for an SAE feature

### Example

* Api Key Authentication (SimpleSecretAuth):

```python
import neuronpedia_inference_client
from neuronpedia_inference_client.models.util_sae_vector_post200_response import UtilSaeVectorPost200Response
from neuronpedia_inference_client.models.util_sae_vector_post_request import UtilSaeVectorPostRequest
from neuronpedia_inference_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /v1
# See configuration.py for a list of all supported configuration parameters.
configuration = neuronpedia_inference_client.Configuration(
    host = "/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: SimpleSecretAuth
configuration.api_key['SimpleSecretAuth'] = os.environ["API_KEY"]

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# configuration.api_key_prefix['SimpleSecretAuth'] = 'Bearer'

# Enter a context with an instance of the API client
with neuronpedia_inference_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = neuronpedia_inference_client.DefaultApi(api_client)
    util_sae_vector_post_request = neuronpedia_inference_client.UtilSaeVectorPostRequest() # UtilSaeVectorPostRequest | 

    try:
        # Get the raw vector for an SAE feature
        api_response = api_instance.util_sae_vector_post(util_sae_vector_post_request)
        print("The response of DefaultApi->util_sae_vector_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->util_sae_vector_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **util_sae_vector_post_request** | [**UtilSaeVectorPostRequest**](UtilSaeVectorPostRequest.md)|  | 

### Return type

[**UtilSaeVectorPost200Response**](UtilSaeVectorPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successfully retrieved SAE vector |  -  |
**401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

