# DefaultApi

All URIs are relative to */v1*

| Method | HTTP request | Description |
|------------- | ------------- | -------------|
| [**activationAllBatchPost**](DefaultApi.md#activationallbatchpostoperation) | **POST** /activation/all-batch | For a given batch of prompts, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options. |
| [**activationAllPost**](DefaultApi.md#activationallpostoperation) | **POST** /activation/all | For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options. |
| [**activationSingleBatchPost**](DefaultApi.md#activationsinglebatchpostoperation) | **POST** /activation/single-batch | Given a batch of text prompts, returns the activation values for a single SAE latent or custom vector+hook. |
| [**activationSinglePost**](DefaultApi.md#activationsinglepostoperation) | **POST** /activation/single | Given a text prompt, returns the activation values for a single SAE latent or custom vector+hook. |
| [**activationSourcePost**](DefaultApi.md#activationsourcepostoperation) | **POST** /activation/source | For a given prompt, get the top activating features for a source (eg 0-gemmascope-res-65k or 5-gemmascope-res-65k), and return the results as a 3D array of prompt x prompt_token x feature_index. |
| [**activationTopkByTokenBatchPost**](DefaultApi.md#activationtopkbytokenbatchpostoperation) | **POST** /activation/topk-by-token-batch | For a given batch of prompts, get the top activating features at each token position for a single SAE. |
| [**activationTopkByTokenPost**](DefaultApi.md#activationtopkbytokenpostoperation) | **POST** /activation/topk-by-token | For a given prompt, get the top activating features at each token position for a single SAE. |
| [**steerCompletionChatPost**](DefaultApi.md#steercompletionchatpostoperation) | **POST** /steer/completion-chat | For a given prompt, complete it by steering with the given feature or vector |
| [**steerCompletionPost**](DefaultApi.md#steercompletionpost) | **POST** /steer/completion | For a given prompt, complete it by steering with the given feature or vector |
| [**tokenizePost**](DefaultApi.md#tokenizepostoperation) | **POST** /tokenize | Tokenize input text for a given model |
| [**utilSaeTopkByDecoderCossimPost**](DefaultApi.md#utilsaetopkbydecodercossimpostoperation) | **POST** /util/sae-topk-by-decoder-cossim | Given a specific vector or SAE feature, return the top features by cosine similarity in the same SAE |
| [**utilSaeVectorPost**](DefaultApi.md#utilsaevectorpostoperation) | **POST** /util/sae-vector | Get the raw vector for an SAE feature |



## activationAllBatchPost

> ActivationAllBatchPost200Response activationAllBatchPost(activationAllBatchPostRequest)

For a given batch of prompts, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationAllBatchPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationAllBatchPostRequest
    activationAllBatchPostRequest: ...,
  } satisfies ActivationAllBatchPostOperationRequest;

  try {
    const data = await api.activationAllBatchPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationAllBatchPostRequest** | [ActivationAllBatchPostRequest](ActivationAllBatchPostRequest.md) |  | |

### Return type

[**ActivationAllBatchPost200Response**](ActivationAllBatchPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## activationAllPost

> ActivationAllPost200Response activationAllPost(activationAllPostRequest)

For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationAllPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationAllPostRequest
    activationAllPostRequest: ...,
  } satisfies ActivationAllPostOperationRequest;

  try {
    const data = await api.activationAllPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationAllPostRequest** | [ActivationAllPostRequest](ActivationAllPostRequest.md) |  | |

### Return type

[**ActivationAllPost200Response**](ActivationAllPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## activationSingleBatchPost

> ActivationSingleBatchPost200Response activationSingleBatchPost(activationSingleBatchPostRequest)

Given a batch of text prompts, returns the activation values for a single SAE latent or custom vector+hook.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationSingleBatchPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationSingleBatchPostRequest
    activationSingleBatchPostRequest: ...,
  } satisfies ActivationSingleBatchPostOperationRequest;

  try {
    const data = await api.activationSingleBatchPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationSingleBatchPostRequest** | [ActivationSingleBatchPostRequest](ActivationSingleBatchPostRequest.md) |  | |

### Return type

[**ActivationSingleBatchPost200Response**](ActivationSingleBatchPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## activationSinglePost

> ActivationSinglePost200Response activationSinglePost(activationSinglePostRequest)

Given a text prompt, returns the activation values for a single SAE latent or custom vector+hook.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationSinglePostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationSinglePostRequest
    activationSinglePostRequest: ...,
  } satisfies ActivationSinglePostOperationRequest;

  try {
    const data = await api.activationSinglePost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationSinglePostRequest** | [ActivationSinglePostRequest](ActivationSinglePostRequest.md) |  | |

### Return type

[**ActivationSinglePost200Response**](ActivationSinglePost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## activationSourcePost

> ActivationSourcePost200Response activationSourcePost(activationSourcePostRequest)

For a given prompt, get the top activating features for a source (eg 0-gemmascope-res-65k or 5-gemmascope-res-65k), and return the results as a 3D array of prompt x prompt_token x feature_index.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationSourcePostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationSourcePostRequest
    activationSourcePostRequest: ...,
  } satisfies ActivationSourcePostOperationRequest;

  try {
    const data = await api.activationSourcePost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationSourcePostRequest** | [ActivationSourcePostRequest](ActivationSourcePostRequest.md) |  | |

### Return type

[**ActivationSourcePost200Response**](ActivationSourcePost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## activationTopkByTokenBatchPost

> ActivationTopkByTokenBatchPost200Response activationTopkByTokenBatchPost(activationTopkByTokenBatchPostRequest)

For a given batch of prompts, get the top activating features at each token position for a single SAE.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationTopkByTokenBatchPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationTopkByTokenBatchPostRequest
    activationTopkByTokenBatchPostRequest: ...,
  } satisfies ActivationTopkByTokenBatchPostOperationRequest;

  try {
    const data = await api.activationTopkByTokenBatchPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationTopkByTokenBatchPostRequest** | [ActivationTopkByTokenBatchPostRequest](ActivationTopkByTokenBatchPostRequest.md) |  | |

### Return type

[**ActivationTopkByTokenBatchPost200Response**](ActivationTopkByTokenBatchPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## activationTopkByTokenPost

> ActivationTopkByTokenPost200Response activationTopkByTokenPost(activationTopkByTokenPostRequest)

For a given prompt, get the top activating features at each token position for a single SAE.

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { ActivationTopkByTokenPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // ActivationTopkByTokenPostRequest
    activationTopkByTokenPostRequest: ...,
  } satisfies ActivationTopkByTokenPostOperationRequest;

  try {
    const data = await api.activationTopkByTokenPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **activationTopkByTokenPostRequest** | [ActivationTopkByTokenPostRequest](ActivationTopkByTokenPostRequest.md) |  | |

### Return type

[**ActivationTopkByTokenPost200Response**](ActivationTopkByTokenPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## steerCompletionChatPost

> SteerCompletionChatPost200Response steerCompletionChatPost(steerCompletionChatPostRequest)

For a given prompt, complete it by steering with the given feature or vector

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { SteerCompletionChatPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // SteerCompletionChatPostRequest
    steerCompletionChatPostRequest: ...,
  } satisfies SteerCompletionChatPostOperationRequest;

  try {
    const data = await api.steerCompletionChatPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **steerCompletionChatPostRequest** | [SteerCompletionChatPostRequest](SteerCompletionChatPostRequest.md) |  | |

### Return type

[**SteerCompletionChatPost200Response**](SteerCompletionChatPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## steerCompletionPost

> SteerCompletionPost200Response steerCompletionPost(steerCompletionRequest)

For a given prompt, complete it by steering with the given feature or vector

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { SteerCompletionPostRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // SteerCompletionRequest
    steerCompletionRequest: ...,
  } satisfies SteerCompletionPostRequest;

  try {
    const data = await api.steerCompletionPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **steerCompletionRequest** | [SteerCompletionRequest](SteerCompletionRequest.md) |  | |

### Return type

[**SteerCompletionPost200Response**](SteerCompletionPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## tokenizePost

> TokenizePost200Response tokenizePost(tokenizePostRequest)

Tokenize input text for a given model

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { TokenizePostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // TokenizePostRequest
    tokenizePostRequest: ...,
  } satisfies TokenizePostOperationRequest;

  try {
    const data = await api.tokenizePost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **tokenizePostRequest** | [TokenizePostRequest](TokenizePostRequest.md) |  | |

### Return type

[**TokenizePost200Response**](TokenizePost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successful tokenization |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## utilSaeTopkByDecoderCossimPost

> UtilSaeTopkByDecoderCossimPost200Response utilSaeTopkByDecoderCossimPost(utilSaeTopkByDecoderCossimPostRequest)

Given a specific vector or SAE feature, return the top features by cosine similarity in the same SAE

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { UtilSaeTopkByDecoderCossimPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // UtilSaeTopkByDecoderCossimPostRequest
    utilSaeTopkByDecoderCossimPostRequest: ...,
  } satisfies UtilSaeTopkByDecoderCossimPostOperationRequest;

  try {
    const data = await api.utilSaeTopkByDecoderCossimPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **utilSaeTopkByDecoderCossimPostRequest** | [UtilSaeTopkByDecoderCossimPostRequest](UtilSaeTopkByDecoderCossimPostRequest.md) |  | |

### Return type

[**UtilSaeTopkByDecoderCossimPost200Response**](UtilSaeTopkByDecoderCossimPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved results |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


## utilSaeVectorPost

> UtilSaeVectorPost200Response utilSaeVectorPost(utilSaeVectorPostRequest)

Get the raw vector for an SAE feature

### Example

```ts
import {
  Configuration,
  DefaultApi,
} from 'neuronpedia-inference-client';
import type { UtilSaeVectorPostOperationRequest } from 'neuronpedia-inference-client';

async function example() {
  console.log("🚀 Testing neuronpedia-inference-client SDK...");
  const config = new Configuration({ 
    // To configure API key authorization: SimpleSecretAuth
    apiKey: "YOUR API KEY",
  });
  const api = new DefaultApi(config);

  const body = {
    // UtilSaeVectorPostRequest
    utilSaeVectorPostRequest: ...,
  } satisfies UtilSaeVectorPostOperationRequest;

  try {
    const data = await api.utilSaeVectorPost(body);
    console.log(data);
  } catch (error) {
    console.error(error);
  }
}

// Run the test
example().catch(console.error);
```

### Parameters


| Name | Type | Description  | Notes |
|------------- | ------------- | ------------- | -------------|
| **utilSaeVectorPostRequest** | [UtilSaeVectorPostRequest](UtilSaeVectorPostRequest.md) |  | |

### Return type

[**UtilSaeVectorPost200Response**](UtilSaeVectorPost200Response.md)

### Authorization

[SimpleSecretAuth](../README.md#SimpleSecretAuth)

### HTTP request headers

- **Content-Type**: `application/json`
- **Accept**: `application/json`


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
| **200** | Successfully retrieved SAE vector |  -  |
| **401** | X-SECRET-KEY header is missing or invalid |  * WWW_Authenticate -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)

