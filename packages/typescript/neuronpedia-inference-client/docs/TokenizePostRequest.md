
# TokenizePostRequest

Tokenize input text for a given model

## Properties

Name | Type
------------ | -------------
`model` | string
`text` | string
`prependBos` | boolean

## Example

```typescript
import type { TokenizePostRequest } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "model": gpt2-small,
  "text": tokenize me! :D,
  "prependBos": null,
} satisfies TokenizePostRequest

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as TokenizePostRequest
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


