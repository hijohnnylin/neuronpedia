
# ActivationSinglePost200Response

Response for NPActivationSingleRequest. Contains the activation values and tokenized prompt.

## Properties

Name | Type
------------ | -------------
`activation` | [ActivationSinglePost200ResponseActivation](ActivationSinglePost200ResponseActivation.md)
`tokens` | Array&lt;string&gt;

## Example

```typescript
import type { ActivationSinglePost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "activation": null,
  "tokens": null,
} satisfies ActivationSinglePost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationSinglePost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


