
# ActivationSourcePost200ResponseResultsInner

One prompt\'s results, only including non-zero values and non-zero activations

## Properties

Name | Type
------------ | -------------
`tokens` | Array&lt;string&gt;
`activeFeatures` | { [key: string]: Array&lt;Array&lt;number&gt;&gt;; }

## Example

```typescript
import type { ActivationSourcePost200ResponseResultsInner } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "tokens": null,
  "activeFeatures": null,
} satisfies ActivationSourcePost200ResponseResultsInner

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationSourcePost200ResponseResultsInner
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


