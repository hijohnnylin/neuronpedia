# Neuronpedia Python Library

## Authentication

Some APIs on Neuronpedia require an API key. For example, if you want to bookmark something in your account, or upload a new vector, you'll need to identify yourself with a Neuronpedia API key.

### Setting the API Key

1. Sign up for free at `neuronpedia.org`.
2. Get your Neuronpedia API key from `neuronpedia.org/account`.
3. Set the environment variable `NEURONPEDIA_API_KEY` to your API key. You can do this through a `.env` file or other similar methods.

### Example: Upload a Vector, then Steer With It

```
from neuronpedia.sample_data import GEMMA2_2B_IT_DINOSAURS_VECTOR
from neuronpedia.np_vector import NPVector
import os, json

# from neuronpedia.org/account
os.environ["NEURONPEDIA_API_KEY"] = "YOUR_NP_API_KEY"

# upload the custom vector
np_vector = NPVector.new(
    label="dinosaurs",
    model_id="gemma-2-2b-it",
    layer_num=20,
    hook_type="hook_resid_pre",
    vector=GEMMA2_2B_IT_DINOSAURS_VECTOR,
    default_steer_strength=44,
)

# steer with it
responseJson = np_vector.steer_chat(
    steered_chat_messages=[{"role": "user", "content": "Write a one sentence story."}]
)

print(json.dumps(responseJson, indent=2))
print("UI Steering at: " + responseJson["shareUrl"])
```

The output of the above will be similar to:

```
{
  "STEERED": {
    "chat_template": [
      {
        "content": "Write a one sentence story.",
        "role": "user"
      },
      {
        "content": "The last dinosaur roared, its breath a smoke-filled mirror of the dying sun.",
        "role": "model"
      }
    ],
    "raw": "<bos><start_of_turn>user\nWrite a one sentence story.<end_of_turn>\n<start_of_turn>model\nThe last dinosaur roared, its breath a smoke-filled mirror of the dying sun. \n<end_of_turn><eos>"
  },
  [...]
}
```

See the `examples` folder for detailed notebooks and usage.

### Important: Uniquely Identifying a Vector/Feature

On Neuronpedia, a vector or feature has a unique identifier comprised of three parts:

- **Model ID:** The ID of the model that this vector belongs to. For example, `gpt2-small`.
- **Source:** A "group name" that starts with the layer number, and usually contains some other identifying information depending on what it is, like number of features in the group. Some examples of source:
  - `6-res-jb`: `6` = Layer 6, `res` = residual stream, `jb` = Joseph Bloom
  - `3-gemmascope-att-16k`: `3` = Layer 3, `gemmascope` = Gemma Scope, `att` = Attention, `16k` = 16k-width SAE
- **Index:** The index in the model + source. This is a string, and currently the string is always an integer. However, this may change in the future.

Example:

- `gemma-2-2b/3-gemmascope-att-16k/4232`
  - Model: gemma-2-2b
  - Source: 3-gemmascope-att-16k
  - Index: 4232

To make API calls that involve an existing vector or feature on Neuronpedia, you pass the model, source, and index as required.

Example API Call: Get SAE Feature (including activations, explanations, etc)

```
from neuronpedia.np_sae_feature import SAEFeature

sae_feature = SAEFeature.get("gemma-2-2b", "3-gemmascope-att-16k", "4232")
print(sae_feature)
```
