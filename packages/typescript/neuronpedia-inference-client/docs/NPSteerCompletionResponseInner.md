
# NPSteerCompletionResponseInner

A streamed steering/default response. Output is either the whole response or a chunk, depending on response type.

## Properties

Name | Type
------------ | -------------
`type` | [NPSteerType](NPSteerType.md)
`output` | string
`logprobs` | [Array&lt;NPLogprob&gt;](NPLogprob.md)

## Example

```typescript
import type { NPSteerCompletionResponseInner } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "type": null,
  "output": null,
  "logprobs": null,
} satisfies NPSteerCompletionResponseInner

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as NPSteerCompletionResponseInner
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


