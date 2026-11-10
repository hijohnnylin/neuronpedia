# Neuron dashboards

These scripts build the MLP neuron dashboards (Qwen3-32B, Qwen3-8B, Qwen3.5-4B, Gemma 4 31B IT,
Gemma 4 12B base and IT, Llama 3.3 70B), and the per-neuron tables that the inference server uses
for neuron search and ablation. The webapp reads
the dashboards from `https://dashboards.neuronpedia.org/v2/<model>/` (R2 key
`neuron-dashboards/v2/<model>/`).

Run them on a GPU pod from one directory, with `creds.sh` beside them. `creds.sh` is not in the
repo. It exports `HF_TOKEN`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and
`R2_S3_ENDPOINT`.

## Dashboards

1. `build_corpus.py`: builds the shared corpus and the token files for each model, once. The shared
   v2 corpus is in R2 already. Do not run `finalize.py --upload-corpus`: it replaces it.
2. `setup.sh MODEL HF_ID TOKENS_FROM [HF_EXCLUDE]`: on each model's pod, makes the venv and
   `env.sh`, and downloads the weights and the corpus. `TOKENS_FROM` names a model in R2 with the
   same tokenizer (Qwen3-8B uses `qwen3-32b`, Gemma 4 12B uses `gemma-4-31b-it`), or `build`.
3. `dash.py --model MODEL --out RUN_DIR --exclude-docs CORPUS/removed_docs.json`: one pass over
   the corpus, for every neuron. It writes a checkpoint now and then and resumes from it.
4. `validate.py RUN_DIR CORPUS MODEL LAYER/NEURON...`: checks known neurons and internal
   consistency.
5. `finalize.py --run RUN_DIR`: writes one gzip record per neuron and uploads everything to R2
   under `neuron-dashboards/v2/`. `verify_r2.py` checks the upload.
6. `build_idx.py MODEL...`: writes the `.idx` offset files the browser uses for Range requests.

`post.sh` (steps 5 and 6, then stops the pod) and `pct.sh` (the percentile table) wait for `dash.py`
to end and then run.

v2 has 7,621 code documents removed for their licenses (`corpus/removed_docs.json`). For Qwen3-32B,
Gemma 4 31B and Llama 3.3 70B, the v1 output was edited to remove them; that step is not in this
folder. Newer models do not read them at all (`--exclude-docs`).

## Tables for the inference server

- `build_act_range_from_records.py OUT_DIR MODEL...`: reads the live `v2` records and writes the
  neuron act range table (max, min and quantiles per neuron) for `NEURON_ACT_RANGE_PATH`. Upload it
  to `neuron-dashboards/v2/<model>/neuron_act_range.safetensors`.
- `validate_act_range.py TABLE MODEL N_LAYERS D_MLP [N]`: checks a table against random records.
- `verify_rank_pod.py POD_URL MODEL LAYERS PROMPT SORT_INDEXES HF_TOKENIZER [N]`: checks a pod's
  search-all order against rarity computed again from the records. Needs `INFERENCE_SECRET`.
- `build_percentiles.py RAW_DIR OUT.safetensors`: the |activation| percentile table for neuron
  ablation (`NEURON_PERCENTILES_PATH`).

## Database exports

`convert_to_export.py MODEL [--layers 0,1] [--out /data/exports]` reads the `v2` records and writes
one export source per layer to `<out>/<model>/neurons/<layer>/`, with the same files as the SAE
exports (`features/`, `activations/`, `model.jsonl`, `sourceset.jsonl`, `source.jsonl`). The rows
are the same as `recordToNeuronDashboard` in the webapp makes. `export_meta.json` holds the model
and source set rows for each model; add a model there before you convert it.
