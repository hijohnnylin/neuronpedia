
# TokenizePost200Response


## Properties

Name | Type
------------ | -------------
`tokens` | Array&lt;number&gt;
`tokenStrings` | Array&lt;string&gt;
`prependBos` | boolean

## Example

```typescript
import type { TokenizePost200Response } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "tokens": null,
  "tokenStrings": null,
  "prependBos": null,
} satisfies TokenizePost200Response

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as TokenizePost200Response
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


