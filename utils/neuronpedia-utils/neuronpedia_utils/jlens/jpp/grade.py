"""Write the model-correctness CSV that fit-weights and evaluate filter on."""

import argparse
import logging

from lens_evals.readout_evals.readout_eval_items import (
    MACRO_EVALS,
    READOUT_EVALS_DATA_DIR,
    load_readout_eval_items,
)
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--hf-model-name", required=True)
parser.add_argument("--layers", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

model = get_hf_model(args.hf_model_name)
layers = [int(x) for x in args.layers.split(",")]
runner = ReadoutEvalRunner(model, {}, layers=layers, max_seq_len=512)
items = load_readout_eval_items(READOUT_EVALS_DATA_DIR, slugs=MACRO_EVALS)
df = runner.grade_model_correctness(items, hf_model_name=args.hf_model_name)
df.to_csv(args.out, index=False)
col = df[args.hf_model_name]
print(
    f"graded {len(df)} items: {int((col == True).sum())} correct, "
    f"{int((col == False).sum())} wrong, {int(col.isna().sum())} ungradeable",
    flush=True,
)
for slug, group in df.groupby("eval"):
    c = group[args.hf_model_name]
    print(
        f"  {slug}: {int((c == True).sum())}/{int(c.notna().sum())} correct, {int(c.isna().sum())} NA",
        flush=True,
    )
