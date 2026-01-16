# we have issues with our cloud provider giving "dud" GPUs, so this is a simple sanity check we use.
# the two non-zero counts should be the same.

from nnsight import LanguageModel
import torch

model_id = "meta-llama/Llama-3.1-8B"
prompt = "Hello world."
layer_num = 5

print("Loading model with device_map='auto'...")
model = LanguageModel(
    model_id,
    device_map="auto", 
    dtype=torch.bfloat16,
    trust_remote_code=True, 
)

with model.trace(prompt):
    outputs = model.model.layers[layer_num].output.save()

non_zero_mask = outputs != 0
non_zero_count = non_zero_mask.sum().item()
print(f"DEVICE MAP: auto, non-zero count in outputs: {non_zero_count}")


print("Reloading model with device_map='cuda:0'...")
model = LanguageModel(
    model_id,
    device_map="cuda:0", 
    dtype=torch.bfloat16,
    trust_remote_code=True, 
)
with model.trace(prompt):
    outputs = model.model.layers[layer_num].output.save()

non_zero_mask = outputs != 0
non_zero_count = non_zero_mask.sum().item()
print(f"DEVICE MAP: cuda:0, non-zero count in outputs: {non_zero_count}")