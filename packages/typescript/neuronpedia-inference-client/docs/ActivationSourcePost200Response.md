
# ActivationSourcePost200Response

All prompts results, only including non-zero features and non-zero activations

## Properties

Name | Type
------------ | -------------
`results` | [Array&lt;ActivationSourcePost200ResponseResultsInner&gt;](ActivationSourcePost200ResponseResultsInner.md)

## Example

```typescript
import type { ActivationSourcePost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "results": null,
} satisfies ActivationSourcePost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationSourcePost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


