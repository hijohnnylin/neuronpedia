
# ActivationSourcePostRequest

For a given prompt, get the top activating features for a source (eg 0-gemmascope-res-65k or 5-gemmascope-res-65k), and return the results as a 3D array of prompt x prompt_token x feature_index.

## Properties

Name | Type
------------ | -------------
`prompts` | Array&lt;string&gt;
`model` | string
`source` | string

## Example

```typescript
import type { ActivationSourcePostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "prompts": null,
  "model": null,
  "source": null,
} satisfies ActivationSourcePostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationSourcePostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


