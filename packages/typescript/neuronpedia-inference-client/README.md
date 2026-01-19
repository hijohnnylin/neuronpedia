# neuronpedia-inference-client@1.10.0

A TypeScript SDK client for the localhost API.

## Usage

First, install the SDK from npm.

```bash
npm install neuronpedia-inference-client --save
```

Next, try it out.


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


## Documentation

### API Endpoints

All URIs are relative to */v1*

| Class | Method | HTTP request | Description
| ----- | ------ | ------------ | -------------
*DefaultApi* | [**activationAllBatchPost**](docs/DefaultApi.md#activationallbatchpostoperation) | **POST** /activation/all-batch | For a given batch of prompts, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.
*DefaultApi* | [**activationAllPost**](docs/DefaultApi.md#activationallpostoperation) | **POST** /activation/all | For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.
*DefaultApi* | [**activationSingleBatchPost**](docs/DefaultApi.md#activationsinglebatchpostoperation) | **POST** /activation/single-batch | Given a batch of text prompts, returns the activation values for a single SAE latent or custom vector+hook.
*DefaultApi* | [**activationSinglePost**](docs/DefaultApi.md#activationsinglepostoperation) | **POST** /activation/single | Given a text prompt, returns the activation values for a single SAE latent or custom vector+hook.
*DefaultApi* | [**activationSourcePost**](docs/DefaultApi.md#activationsourcepostoperation) | **POST** /activation/source | For a given prompt, get the top activating features for a source (eg 0-gemmascope-res-65k or 5-gemmascope-res-65k), and return the results as a 3D array of prompt x prompt_token x feature_index.
*DefaultApi* | [**activationTopkByTokenBatchPost**](docs/DefaultApi.md#activationtopkbytokenbatchpostoperation) | **POST** /activation/topk-by-token-batch | For a given batch of prompts, get the top activating features at each token position for a single SAE.
*DefaultApi* | [**activationTopkByTokenPost**](docs/DefaultApi.md#activationtopkbytokenpostoperation) | **POST** /activation/topk-by-token | For a given prompt, get the top activating features at each token position for a single SAE.
*DefaultApi* | [**steerCompletionChatPost**](docs/DefaultApi.md#steercompletionchatpostoperation) | **POST** /steer/completion-chat | For a given prompt, complete it by steering with the given feature or vector
*DefaultApi* | [**steerCompletionPost**](docs/DefaultApi.md#steercompletionpost) | **POST** /steer/completion | For a given prompt, complete it by steering with the given feature or vector
*DefaultApi* | [**tokenizePost**](docs/DefaultApi.md#tokenizepostoperation) | **POST** /tokenize | Tokenize input text for a given model
*DefaultApi* | [**utilSaeTopkByDecoderCossimPost**](docs/DefaultApi.md#utilsaetopkbydecodercossimpostoperation) | **POST** /util/sae-topk-by-decoder-cossim | Given a specific vector or SAE feature, return the top features by cosine similarity in the same SAE
*DefaultApi* | [**utilSaeVectorPost**](docs/DefaultApi.md#utilsaevectorpostoperation) | **POST** /util/sae-vector | Get the raw vector for an SAE feature


### Models

- [ActivationAllBatchPost200Response](docs/ActivationAllBatchPost200Response.md)
- [ActivationAllBatchPost200ResponseResultsInner](docs/ActivationAllBatchPost200ResponseResultsInner.md)
- [ActivationAllBatchPostRequest](docs/ActivationAllBatchPostRequest.md)
- [ActivationAllPost200Response](docs/ActivationAllPost200Response.md)
- [ActivationAllPost200ResponseActivationsInner](docs/ActivationAllPost200ResponseActivationsInner.md)
- [ActivationAllPostRequest](docs/ActivationAllPostRequest.md)
- [ActivationSingleBatchPost200Response](docs/ActivationSingleBatchPost200Response.md)
- [ActivationSingleBatchPost200ResponseResultsInner](docs/ActivationSingleBatchPost200ResponseResultsInner.md)
- [ActivationSingleBatchPostRequest](docs/ActivationSingleBatchPostRequest.md)
- [ActivationSinglePost200Response](docs/ActivationSinglePost200Response.md)
- [ActivationSinglePost200ResponseActivation](docs/ActivationSinglePost200ResponseActivation.md)
- [ActivationSinglePostRequest](docs/ActivationSinglePostRequest.md)
- [ActivationSourcePost200Response](docs/ActivationSourcePost200Response.md)
- [ActivationSourcePost200ResponseResultsInner](docs/ActivationSourcePost200ResponseResultsInner.md)
- [ActivationSourcePostRequest](docs/ActivationSourcePostRequest.md)
- [ActivationTopkByTokenBatchPost200Response](docs/ActivationTopkByTokenBatchPost200Response.md)
- [ActivationTopkByTokenBatchPost200ResponseResultsInner](docs/ActivationTopkByTokenBatchPost200ResponseResultsInner.md)
- [ActivationTopkByTokenBatchPostRequest](docs/ActivationTopkByTokenBatchPostRequest.md)
- [ActivationTopkByTokenPost200Response](docs/ActivationTopkByTokenPost200Response.md)
- [ActivationTopkByTokenPost200ResponseResultsInner](docs/ActivationTopkByTokenPost200ResponseResultsInner.md)
- [ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner](docs/ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner.md)
- [ActivationTopkByTokenPostRequest](docs/ActivationTopkByTokenPostRequest.md)
- [NPFeature](docs/NPFeature.md)
- [NPLogprob](docs/NPLogprob.md)
- [NPLogprobTop](docs/NPLogprobTop.md)
- [NPSteerChatMessage](docs/NPSteerChatMessage.md)
- [NPSteerChatResult](docs/NPSteerChatResult.md)
- [NPSteerCompletionResponseInner](docs/NPSteerCompletionResponseInner.md)
- [NPSteerFeature](docs/NPSteerFeature.md)
- [NPSteerMethod](docs/NPSteerMethod.md)
- [NPSteerType](docs/NPSteerType.md)
- [NPSteerVector](docs/NPSteerVector.md)
- [SteerCompletionChatPost200Response](docs/SteerCompletionChatPost200Response.md)
- [SteerCompletionChatPost200ResponseAssistantAxisInner](docs/SteerCompletionChatPost200ResponseAssistantAxisInner.md)
- [SteerCompletionChatPost200ResponseAssistantAxisInnerTurnsInner](docs/SteerCompletionChatPost200ResponseAssistantAxisInnerTurnsInner.md)
- [SteerCompletionChatPostRequest](docs/SteerCompletionChatPostRequest.md)
- [SteerCompletionPost200Response](docs/SteerCompletionPost200Response.md)
- [SteerCompletionRequest](docs/SteerCompletionRequest.md)
- [TokenizePost200Response](docs/TokenizePost200Response.md)
- [TokenizePostRequest](docs/TokenizePostRequest.md)
- [UtilSaeTopkByDecoderCossimPost200Response](docs/UtilSaeTopkByDecoderCossimPost200Response.md)
- [UtilSaeTopkByDecoderCossimPost200ResponseTopkDecoderCossimFeaturesInner](docs/UtilSaeTopkByDecoderCossimPost200ResponseTopkDecoderCossimFeaturesInner.md)
- [UtilSaeTopkByDecoderCossimPostRequest](docs/UtilSaeTopkByDecoderCossimPostRequest.md)
- [UtilSaeVectorPost200Response](docs/UtilSaeVectorPost200Response.md)
- [UtilSaeVectorPostRequest](docs/UtilSaeVectorPostRequest.md)

### Authorization


Authentication schemes defined for the API:
<a id="SimpleSecretAuth"></a>
#### SimpleSecretAuth


- **Type**: API key
- **API key parameter name**: `X-SECRET-KEY`
- **Location**: HTTP header

## About

This TypeScript SDK client supports the [Fetch API](https://fetch.spec.whatwg.org/)
and is automatically generated by the
[OpenAPI Generator](https://openapi-generator.tech) project:

- API version: `1.9.0`
- Package version: `1.10.0`
- Generator version: `7.18.0`
- Build package: `org.openapitools.codegen.languages.TypeScriptFetchClientCodegen`

The generated npm module supports the following:

- Environments
  * Node.js
  * Webpack
  * Browserify
- Language levels
  * ES5 - you must have a Promises/A+ library installed
  * ES6
- Module systems
  * CommonJS
  * ES6 module system


## Development

### Building

To build the TypeScript source code, you need to have Node.js and npm installed.
After cloning the repository, navigate to the project directory and run:

```bash
npm install
npm run build
```

### Publishing

Once you've built the package, you can publish it to npm:

```bash
npm publish
```

## License

[MIT]()
