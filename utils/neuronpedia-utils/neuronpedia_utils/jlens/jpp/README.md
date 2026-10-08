# jpp — J++ Lens fitting and conversion for Neuronpedia

Fits a J++ Lens with the recipe from *J++ Lens: Jacobian Filtering Enables More Faithful
Workspace Lenses* (Ayonrinde & Lindsey, 2026), and writes J++ Lens files in the format the
inference server reads.

> The fit runs the paper's code, [`safety-research/jpp_lens`](https://github.com/safety-research/jpp_lens)
> (Apache-2.0). These scripts call it; they do not contain it. You need a checkout of it to fit
> a lens. You do not need it to convert one.

## Layout

| Path | Purpose |
| --- | --- |
| `fit-jpp-lens.sh` | Fit and score one model: grade, router, expert shards, merge, expert weights, eval. |
| `jpp_cli_gpu.py` | The paper's `scripts/jpp_cli.py`, with the router's PCA and k-means on the GPU. |
| `grade.py` | Write the model-correctness CSV that `fit-weights` and `evaluate` filter on. |
| `convert-jpp-lens.py` | Write a fitted J++ Lens (ours or `koayon/jpp-lenses`) in the Neuronpedia format, with a `config.yaml`. |
| `dequant_mlx.py` | Write an HF copy of a base model with MLX 4-bit weights put back as bf16, to score a lens on a quantized model. |

## Fit a lens

```bash
git clone https://github.com/safety-research/jpp_lens jpp_lens && (cd jpp_lens && uv sync)
JPP_LENS_DIR=jpp_lens JPP_OUT_DIR=jpp-fits ./fit-jpp-lens.sh \
  Qwen/Qwen3.5-2B qwen3.5-2b 96 \
  qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt
```

The recipe fits the router on 1,000 WikiText-103 records and the experts on `<n-prompts>`
records (96 for our small Qwen lenses; the paper uses 64), with bf16 base models. The last
argument is optional. With it, the eval also scores our Jacobian lens for that model.

Each step is skipped when its output exists, so you can run the command again after a stop.
See the comment at the top of `fit-jpp-lens.sh` for the environment settings.

**GatedDeltaNet models (Qwen3.5, Qwen3.6):** `jpp_cli_gpu.py` blocks the `fla` kernels for
`fit-shard`, because the LRP backward pass needs the torch GatedDeltaNet path.

## Convert a lens

```bash
hf download koayon/jpp-lenses qwen3.5-9b/lens.pt --local-dir /tmp/koayon
uv run python jpp/convert-jpp-lens.py /tmp/koayon/qwen3.5-9b/lens.pt \
  --np-model-id qwen3.5-9b --source "koayon/jpp-lenses@<sha>:qwen3.5-9b/lens.pt"
```

This writes:

```
exports/<np_model_id>/jpp/Salesforce-wikitext/
  <model>_jpp_lens.pt   # J: {layer: fp16 [d_model, d_model]}, source_layers, d_model, n_prompts, provenance
  config.yaml           # the fit config, the source, the command, attribution
```

The script changes the key names and casts the fp32 maps to fp16. It stops if a map overflows
fp16, if the relative error is above `2e-3` (the bf16 cast the server does is about
`1.7e-3`), or if the layers and width do not agree with the
model's `config.json`. The maps cover every block but the final one, which is the target.

**DeepSeek-V4-Flash** has 4 residual streams, and the server refuses a lens that does not say
which activation it reads. The J++ fit reads the mean over the streams
(`workspace_lens/residual_streams.py`), so pass:

```bash
--stream-reduce mean --derived-from "safety-research/jpp_lens workspace_lens/residual_streams.py"
```

The server loads `<np_model_id>/jpp/<JPP_DATASET>/<slug>_jpp_lens.pt` from `JPP_HF_REPO`
(default `neuronpedia/jacobian-lens`) when `JPP_LENS=true`.
