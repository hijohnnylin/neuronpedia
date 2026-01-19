
# NPFeature

A feature in Neuronpedia, identified by model, source, and index.

## Properties

Name | Type
------------ | -------------
`model` | string
`source` | string
`index` | number

## Example

```typescript
import type { NPFeature } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "model": gpt2-small,
  "source": 0-res-jb,
  "index": 14057,
} satisfies NPFeature

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as NPFeature
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


