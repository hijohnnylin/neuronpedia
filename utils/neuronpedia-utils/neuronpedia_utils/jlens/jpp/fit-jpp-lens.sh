#!/usr/bin/env bash
# Fit and score one J++ Lens with the safety-research/jpp_lens recipe.
#
# Usage: fit-jpp-lens.sh <hf-model> <slug> <n-prompts> [<jacobian-lens path for the eval>]
#   fit-jpp-lens.sh Qwen/Qwen3.5-2B qwen3.5-2b 96 \
#     qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt
#
# The 4th argument is a file in neuronpedia/jacobian-lens. With it, the eval also scores
# that Jacobian lens, so the two can be compared. Each step is skipped when its output
# exists, so a stopped run continues when you run it again.
#
# Environment:
#   JPP_LENS_DIR       safety-research/jpp_lens checkout with its .venv (default: ./jpp_lens)
#   JPP_OUT_DIR        output root; the run writes <JPP_OUT_DIR>/<slug> (default: ./jpp-fits)
#   JPP_ROUTER_ON_GPU  1 = fit the router on the GPU (default), 0 = the paper's CPU path
#   JPP_ROWS_PER_PASS  Jacobian rows per backward pass (default: the recipe's 16).
#                      64 is faster on an H200 for models up to 4B.
#   OMP_NUM_THREADS    torch CPU threads. Set 4 to 8 on a pod with a CPU quota.
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,7p "$0"; exit 1; }
HF=$1; SLUG=$2; NP=$3; JLENS=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
JPP_LENS_DIR=$(cd "${JPP_LENS_DIR:-./jpp_lens}" && pwd)
OUT=$(mkdir -p "${JPP_OUT_DIR:-./jpp-fits}" && cd "${JPP_OUT_DIR:-./jpp-fits}" && pwd)/$SLUG
mkdir -p "$OUT"
cd "$JPP_LENS_DIR"
export PYTHONPATH=src
PY=.venv/bin/python
if [ "${JPP_ROUTER_ON_GPU:-1}" = 1 ]; then CLI="$PY $HERE/jpp_cli_gpu.py"; else CLI="$PY scripts/jpp_cli.py"; fi
ROWS=${JPP_ROWS_PER_PASS:+--jacobian-rows-per-pass $JPP_ROWS_PER_PASS}
stamp() { echo "=== $(date -u +%H:%M:%S) $SLUG $*"; }

# Every block but the final one, which is the transport target.
LAYERS=$($PY - "$HF" <<'EOF'
import json, sys
from huggingface_hub import hf_hub_download
cfg = json.load(open(hf_hub_download(sys.argv[1], "config.json")))
n = int((cfg.get("text_config") or cfg)["num_hidden_layers"])
print(",".join(str(i) for i in range(n - 1)))
EOF
)

stamp grade
[ -f "$OUT/correctness.csv" ] || $PY "$HERE/grade.py" --hf-model-name "$HF" --layers "$LAYERS" --out "$OUT/correctness.csv"

stamp fit-router
[ -f "$OUT/router.pt" ] || $CLI fit-router --hf-model-name "$HF" --layers "$LAYERS" --out "$OUT/router.pt"

stamp fit-shard n="$NP"
if [ ! -f "$OUT/shard_dir.txt" ]; then
  # shellcheck disable=SC2086
  $CLI fit-shard --hf-model-name "$HF" --layers "$LAYERS" --router-path "$OUT/router.pt" \
    --num-prompts "$NP" $ROWS --artifacts-base-dir "$OUT/artifacts" --checkpoint-name "${SLUG}_experts" \
    | tail -1 > "$OUT/shard_dir.txt"
fi

stamp merge-experts
[ -f "$OUT/experts.pt" ] || $CLI merge-experts --checkpoint-dirs "$(cat "$OUT/shard_dir.txt")" \
  --checkpoint-name "${SLUG}_experts" --out "$OUT/experts.pt" --pooled-lens-out "$OUT/pooled_lens.pt"

stamp fit-weights
[ -f "$OUT/jpp_lens.pt" ] || $CLI fit-weights --hf-model-name "$HF" --layers "$LAYERS" --experts "$OUT/experts.pt" \
  --correctness-csv "$OUT/correctness.csv" --cache-path "$OUT/readout_cache.pt" --out "$OUT/jpp_lens.pt"

stamp evaluate
LENSES=("$OUT/jpp_lens.pt" "$OUT/pooled_lens.pt")
if [ -n "$JLENS" ]; then
  LENSES+=("$($PY -c "from huggingface_hub import hf_hub_download as d; print(d('neuronpedia/jacobian-lens', '$JLENS'))")")
fi
$CLI evaluate --hf-model-name "$HF" --layers "$LAYERS" --correctness-csv "$OUT/correctness.csv" \
  --lens "${LENSES[@]}" logit --out "$OUT/eval_ranks.csv"
cat "$OUT/eval_ranks_pass_at_k.csv"
stamp done
echo "Next: uv run python $HERE/convert-jpp-lens.py $OUT/jpp_lens.pt --np-model-id <id> --source '<source>'"
