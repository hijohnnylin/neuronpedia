
# ActivationTopkByTokenPost200ResponseResultsInner

One token\'s TopK result, including its top features.

## Properties

Name | Type
------------ | -------------
`tokenPosition` | number
`token` | string
`topFeatures` | [Array&lt;ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner&gt;](ActivationTopkByTokenPost200ResponseResultsInnerTopFeaturesInner.md)

## Example

```typescript
import type { ActivationTopkByTokenPost200ResponseResultsInner } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "tokenPosition": null,
  "token": null,
  "topFeatures": null,
} satisfies ActivationTopkByTokenPost200ResponseResultsInner

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationTopkByTokenPost200ResponseResultsInner
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


