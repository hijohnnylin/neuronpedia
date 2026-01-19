
# SteerCompletionChatPost200Response

The steering/default chat responses.

## Properties

Name | Type
------------ | -------------
`assistantAxis` | [Array&lt;SteerCompletionChatPost200ResponseAssistantAxisInner&gt;](SteerCompletionChatPost200ResponseAssistantAxisInner.md)
`outputs` | [Array&lt;NPSteerChatResult&gt;](NPSteerChatResult.md)
`input` | [NPSteerChatResult](NPSteerChatResult.md)

## Example

```typescript
import type { SteerCompletionChatPost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "assistantAxis": null,
  "outputs": null,
  "input": null,
} satisfies SteerCompletionChatPost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as SteerCompletionChatPost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


