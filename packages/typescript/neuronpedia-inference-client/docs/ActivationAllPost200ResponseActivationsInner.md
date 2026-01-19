
# ActivationAllPost200ResponseActivationsInner

One feature and its activation in an NPActivationAllResponse

## Properties

Name | Type
------------ | -------------
`source` | string
`index` | number
`values` | Array&lt;number&gt;
`sumValues` | number
`maxValue` | number
`maxValueIndex` | number
`dfaValues` | Array&lt;number&gt;
`dfaTargetIndex` | number
`dfaMaxValue` | number

## Example

```typescript
import type { ActivationAllPost200ResponseActivationsInner } from 'neuronpedia-inference-client'

// TODO: Update the object below with actual values
const example = {
  "source": null,
  "index": null,
  "values": null,
  "sumValues": null,
  "maxValue": null,
  "maxValueIndex": null,
  "dfaValues": null,
  "dfaTargetIndex": null,
  "dfaMaxValue": null,
} satisfies ActivationAllPost200ResponseActivationsInner

console.log(example)

// Convert the instance to a JSON string
const exampleJSON: string = JSON.stringify(example)
console.log(exampleJSON)

// Parse the JSON string back to an object
const exampleParsed = JSON.parse(exampleJSON) as ActivationAllPost200ResponseActivationsInner
console.log(exampleParsed)
```

[[Back to top]](#) [[Back to API list]](../README.md#api-endpoints) [[Back to Model list]](../README.md#models) [[Back to README]](../README.md)


