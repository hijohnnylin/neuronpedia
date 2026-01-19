
# UtilSaeTopkByDecoderCossimPostRequest


## Properties

Name | Type
------------ | -------------
`feature` | [NPFeature](NPFeature.md)
`vector` | Array&lt;number&gt;
`model` | string
`source` | string
`numResults` | number

## Example

```typescript
import type { UtilSaeTopkByDecoderCossimPostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "feature": null,
  "vector": null,
  "model": null,
  "source": null,
  "numResults": null,
} satisfies UtilSaeTopkByDecoderCossimPostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as UtilSaeTopkByDecoderCossimPostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


