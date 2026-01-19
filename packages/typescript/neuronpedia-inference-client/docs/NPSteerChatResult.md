
# NPSteerChatResult

The formatted and unformatted (\"raw\") chat messages

## Properties

Name | Type
------------ | -------------
`chatTemplate` | [Array&lt;NPSteerChatMessage&gt;](NPSteerChatMessage.md)
`raw` | string
`type` | [NPSteerType](NPSteerType.md)
`logprobs` | [Array&lt;NPLogprob&gt;](NPLogprob.md)

## Example

```typescript
import type { NPSteerChatResult } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "chatTemplate": null,
  "raw": null,
  "type": null,
  "logprobs": null,
} satisfies NPSteerChatResult

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as NPSteerChatResult
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


