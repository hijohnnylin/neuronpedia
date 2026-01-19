
# ActivationSinglePostRequest

Get activations for either a specific feature in an SAE (specified by \"source\" + \"index\") or a custom vector (specified by \"vector\" + \"hook\")

## Properties

Name | Type
------------ | -------------
`prompt` | string
`model` | string
`source` | string
`index` | string
`vector` | Array&lt;number&gt;
`hook` | string

## Example

```typescript
import type { ActivationSinglePostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "prompt": null,
  "model": null,
  "source": null,
  "index": null,
  "vector": null,
  "hook": null,
} satisfies ActivationSinglePostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationSinglePostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


