
# NPSteerVector

A raw vector for steering, including its hook and strength

## Properties

Name | Type
------------ | -------------
`steeringVector` | Array&lt;number&gt;
`strength` | number
`hook` | string

## Example

```typescript
import type { NPSteerVector } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "steeringVector": null,
  "strength": null,
  "hook": blocks.0.hook_resid_pre,
} satisfies NPSteerVector

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as NPSteerVector
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


