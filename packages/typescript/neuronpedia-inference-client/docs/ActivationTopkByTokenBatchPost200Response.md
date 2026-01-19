
# ActivationTopkByTokenBatchPost200Response

Response for NPActivationTopkByTokenBatchRequest. Contains the batch results of top features at each token position and the tokenized prompts.

## Properties

Name | Type
------------ | -------------
`results` | [Array&lt;ActivationTopkByTokenBatchPost200ResponseResultsInner&gt;](ActivationTopkByTokenBatchPost200ResponseResultsInner.md)

## Example

```typescript
import type { ActivationTopkByTokenBatchPost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "results": null,
} satisfies ActivationTopkByTokenBatchPost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationTopkByTokenBatchPost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


