# icalens — ICA Lens as Neuronpedia sources

[ICA Lens](https://arxiv.org/abs/2606.11722) (Liu & Han et al.) fits FastICA to the
residual stream of each layer. Each component is one direction. This folder converts an
ICA Lens artifact (format v4, as published under
[sida/ica-lens](https://huggingface.co/collections/sida/ica-lens)) into SAELens
`StandardSAE` folders. Inference, SAEDashboard and the webapp then treat each layer as a
normal SAE source.

## Conversion

`convert_icalens_to_saelens.py` writes one folder per layer, `blocks.L.hook_resid_post/`,
with `cfg.json`, `sae_weights.safetensors` and `icalens_components.json`.

- `W_enc = (s * n * R).T`, `b_enc = 0`, `W_dec = (W * s / n).T`, `b_dec = center`.
- `s` is +1 or -1 from the profile `tail_direction`. ReLU keeps the interpretable tail.
  Without it, about half of the components show the wrong tail.
- `n` is the norm of each writing column. `W_dec` rows are unit norm, so
  `fold_W_dec_norm()` has no effect. An activation is the norm of the component's write
  into the residual stream. `activation / n` is the standardized ICA score.
- The script checks each layer against the lens before and after a save and load.

```bash
uv run python convert_icalens_to_saelens.py \
  --source-repo sida/icalens-gpt2-small-pile10k \
  --np-model-id gpt2-small --hf-model-id openai-community/gpt2 --model-name gpt2-small \
  --np-set-name icalens-res --upload-repo neuronpedia/icalens-gpt2-small --upload
```

`--model-name` goes into `cfg.json`. Use the TransformerLens name when TransformerLens
supports the model (SAEDashboard needs it), and the Hugging Face id otherwise.

Output goes to `exports/` (gitignored). The script also writes
`exports/<repo>.registry.yaml`. Add that entry to
`apps/inference/neuronpedia_inference/extra_saes.yaml` so inference can load the repo.

## Framing

The lens is only valid for the framing it was fitted with. Inference must match it.

| Model family | Fit framing | Inference |
| --- | --- | --- |
| GPT-2 | `<\|endoftext|>` prefix, excluded | prepends BOS, which is `<\|endoftext|>` |
| Gemma 2 | BOS, excluded | prepends BOS |
| Qwen3.5 Base | `<\|endoftext|>` prefix, excluded | `bos_token: <\|endoftext\|>` in the registry makes it the BOS |
| Chat models | native chat template | prompt as given |

The converter reads the fit prefix from the lens. When the tokenizer has no BOS, it writes
that prefix as `bos_token` in the registry entry. A server cannot serve sets that disagree
on `bos_token`, for example Qwen3.5 ICA Lens and Qwen-Scope.

For dashboards, pass `--no-prepend-bos --prefix-str "<|endoftext|>"` to SAEDashboard for
Qwen3.5 Base. Without `--no-prepend-bos`, it moves the first text token before the prefix.
