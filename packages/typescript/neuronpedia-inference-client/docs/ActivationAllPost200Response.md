
# ActivationAllPost200Response

Response for NPActivationAllRequest. Contains activations for each top feature and the tokenized prompt.

## Properties

Name | Type
------------ | -------------
`activations` | [Array&lt;ActivationAllPost200ResponseActivationsInner&gt;](ActivationAllPost200ResponseActivationsInner.md)
`tokens` | Array&lt;string&gt;
`counts` | Array&lt;Array&lt;number&gt;&gt;

## Example

```typescript
import type { ActivationAllPost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "activations": null,
  "tokens": null,
  "counts": null,
} satisfies ActivationAllPost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationAllPost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


