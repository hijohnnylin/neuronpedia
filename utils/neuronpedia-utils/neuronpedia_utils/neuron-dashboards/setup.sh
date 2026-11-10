#!/bin/bash
# Set up a dashboard pod: venv, weights, corpus. Log: /root/setup.log
# Usage: setup.sh MODEL HF_ID TOKENS_FROM [HF_EXCLUDE]
#   TOKENS_FROM: a model in R2 with the same tokenizer, whose token files we copy;
#                "build" to tokenize here; "local" if the files are already in place.
#   HF_EXCLUDE:  glob of weight files not to download (e.g. "original/*").
set -euo pipefail
MODEL=$1
HF_ID=$2
TOKENS_FROM=$3
HF_EXCLUDE=${4:-}
cd /root
command -v uv > /dev/null || pip install -q uv
uv venv -q /root/venv --python 3.12
cat > /root/env.sh << 'EOF'
. /root/venv/bin/activate
. /root/creds.sh
export HF_HOME=/root/hf
export HF_XET_HIGH_PERFORMANCE=1
EOF
. /root/env.sh
uv pip install -q torch==2.14.1 torchvision torchaudio transformers==5.20.1 accelerate==1.15.0 \
  flash-linear-attention==0.6.0 safetensors pyarrow datasets orjson boto3 huggingface_hub hf_xet numpy ninja
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "$(date) venv ready"

if [ -n "$HF_EXCLUDE" ]; then
  hf download "$HF_ID" --exclude "$HF_EXCLUDE" > /root/dl.log 2>&1
else
  hf download "$HF_ID" > /root/dl.log 2>&1
fi
echo "$(date) weights ready: $(du -sh /root/hf | cut -f1)"

mkdir -p /workspace/corpus/"$MODEL" /workspace/nd
python - "$MODEL" "$TOKENS_FROM" << 'EOF'
import sys
import finalize
model, src = sys.argv[1], sys.argv[2]
c = finalize.r2()
for f in ("docs.parquet", "stats.json", "removed_docs.json"):
    c.download_file(finalize.BUCKET, f"{finalize.PREFIX}/corpus/{f}", f"/workspace/corpus/{f}")
if src not in ("build", "local"):
    for f in ("tokens.npy", "offsets.npy", "vocab.json"):
        c.download_file(finalize.BUCKET, f"{finalize.PREFIX}/{src}/corpus/{f}", f"/workspace/corpus/{model}/{f}")
print("corpus downloaded")
EOF
if [ "$TOKENS_FROM" = build ]; then
  python build_corpus.py --out /workspace/corpus --tokens-only "$MODEL"
fi
ls -la /workspace/corpus/"$MODEL"
echo "$(date) SETUP DONE"
