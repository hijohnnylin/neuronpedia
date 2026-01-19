
# ActivationTopkByTokenBatchPostRequest

Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")

## Properties

Name | Type
------------ | -------------
`prompts` | Array&lt;string&gt;
`model` | string
`source` | string
`topK` | number
`ignoreBos` | boolean

## Example

```typescript
import type { ActivationTopkByTokenBatchPostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "prompts": null,
  "model": null,
  "source": null,
  "topK": null,
  "ignoreBos": null,
} satisfies ActivationTopkByTokenBatchPostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationTopkByTokenBatchPostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


