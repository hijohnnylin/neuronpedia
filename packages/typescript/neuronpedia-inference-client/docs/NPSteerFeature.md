
# NPSteerFeature


## Properties

Name | Type
------------ | -------------
`model` | string
`source` | string
`index` | number
`strength` | number
`steeringVector` | Array&lt;number&gt;

## Example

```typescript
import type { NPSteerFeature } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "model": gpt2-small,
  "source": 0-res-jb,
  "index": 14057,
  "strength": 5,
  "steeringVector": null,
} satisfies NPSteerFeature

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as NPSteerFeature
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


