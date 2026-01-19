
# ActivationAllBatchPostRequest

For a given batch of prompts, get the top activating features for a set of SAEs (eg gemmascope-res-65k), or specific SAEs in the set of SAEs (eg 0-gemmascope-res-65k, 5-gemmascope-res-65k). Also has other customization options.

## Properties

Name | Type
------------ | -------------
`prompts` | Array&lt;string&gt;
`model` | string
`sourceSet` | string
`selectedSources` | Array&lt;string&gt;
`sortByTokenIndexes` | Array&lt;number&gt;
`ignoreBos` | boolean
`featureFilter` | Array&lt;number&gt;
`numResults` | number

## Example

```typescript
import type { ActivationAllBatchPostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "prompts": null,
  "model": null,
  "sourceSet": null,
  "selectedSources": null,
  "sortByTokenIndexes": null,
  "ignoreBos": null,
  "featureFilter": null,
  "numResults": null,
} satisfies ActivationAllBatchPostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationAllBatchPostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


