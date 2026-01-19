
# ActivationAllBatchPost200Response

Response for NPActivationAllBatchRequest. Contains the batch results of activations for each top feature and the tokenized prompts.

## Properties

Name | Type
------------ | -------------
`results` | [Array&lt;ActivationAllBatchPost200ResponseResultsInner&gt;](ActivationAllBatchPost200ResponseResultsInner.md)

## Example

```typescript
import type { ActivationAllBatchPost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "results": null,
} satisfies ActivationAllBatchPost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationAllBatchPost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


