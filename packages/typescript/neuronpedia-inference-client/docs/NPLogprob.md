
# NPLogprob

Logprobs for a single token

## Properties

Name | Type
------------ | -------------
`token` | string
`logprob` | number
`topLogprobs` | [Array&lt;NPLogprobTop&gt;](NPLogprobTop.md)

## Example

```typescript
import type { NPLogprob } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "token": null,
  "logprob": null,
  "topLogprobs": null,
} satisfies NPLogprob

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as NPLogprob
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


