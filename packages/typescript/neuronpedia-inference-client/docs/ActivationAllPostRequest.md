
# ActivationAllPostRequest

For a given prompt, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.

## Properties

Name | Type
------------ | -------------
`prompt` | string
`model` | string
`sourceSet` | string
`selectedSources` | Array&lt;string&gt;
`sortByTokenIndexes` | Array&lt;number&gt;
`ignoreBos` | boolean
`featureFilter` | Array&lt;number&gt;
`numResults` | number

## Example

```typescript
import type { ActivationAllPostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "prompt": null,
  "model": null,
  "sourceSet": null,
  "selectedSources": null,
  "sortByTokenIndexes": null,
  "ignoreBos": null,
  "featureFilter": null,
  "numResults": null,
} satisfies ActivationAllPostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationAllPostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


