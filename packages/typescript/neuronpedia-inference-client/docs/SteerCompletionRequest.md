
# SteerCompletionRequest

Base request for steering

## Properties

Name | Type
------------ | -------------
`prompt` | string
`model` | string
`steerMethod` | [NPSteerMethod](NPSteerMethod.md)
`normalizeSteering` | boolean
`types` | [Array&lt;NPSteerType&gt;](NPSteerType.md)
`features` | [Array&lt;NPSteerFeature&gt;](NPSteerFeature.md)
`vectors` | [Array&lt;NPSteerVector&gt;](NPSteerVector.md)
`nCompletionTokens` | number
`temperature` | number
`strengthMultiplier` | number
`freqPenalty` | number
`seed` | number
`stream` | boolean
`nLogprobs` | number

## Example

```typescript
import type { SteerCompletionRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "prompt": null,
  "model": null,
  "steerMethod": null,
  "normalizeSteering": null,
  "types": null,
  "features": null,
  "vectors": null,
  "nCompletionTokens": null,
  "temperature": null,
  "strengthMultiplier": null,
  "freqPenalty": null,
  "seed": null,
  "stream": null,
  "nLogprobs": null,
} satisfies SteerCompletionRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as SteerCompletionRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


