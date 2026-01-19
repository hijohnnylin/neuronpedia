
# ActivationSingleBatchPost200Response

Response for NPActivationBatchRequest. Contains the batch results of activation values and tokenized prompt.

## Properties

Name | Type
------------ | -------------
`results` | [Array&lt;ActivationSingleBatchPost200ResponseResultsInner&gt;](ActivationSingleBatchPost200ResponseResultsInner.md)

## Example

```typescript
import type { ActivationSingleBatchPost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "results": null,
} satisfies ActivationSingleBatchPost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationSingleBatchPost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


