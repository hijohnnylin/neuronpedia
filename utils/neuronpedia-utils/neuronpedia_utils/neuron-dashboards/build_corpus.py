"""Build the shared text corpus and the per-model token files.

Quotas are in Qwen3 tokens; each document is cut to its first 1024 Qwen3 tokens, and the stored
text is cut at the same place, so both models see the same text. Sources:
  70% FineWeb sample-10BT (English web), 15% FineWeb-2 (10 languages),
  10% github-code-clean (code), 5% FineMath-4plus (math).

Writes, under OUT (default /workspace/corpus):
  docs.parquet               doc_id, source, text
  <model>/tokens.npy         uint32, all documents concatenated
  <model>/offsets.npy        int64, n_docs + 1
  <model>/vocab.json         decoded string for each token id
  stats.json                 per-source document and token counts
Gemma 4 and Llama get a BOS token at position 0 and are cut to 1024 tokens in total.
--tokens-only MODEL reads an existing docs.parquet and writes only that model's token files.
A model with the same tokenizer as one above (Qwen3-8B, Gemma 4 12B) uses that model's files.
"""

import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_TOK = 1024
MIN_TOK = 64
TOKENIZERS = {
    "qwen3-32b": "Qwen/Qwen3-32B",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "llama3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
}
FW2_LANGS = {
    "cmn_Hani": 0.27,
    "jpn_Jpan": 0.10,
    "kor_Hang": 0.07,
    "rus_Cyrl": 0.10,
    "spa_Latn": 0.10,
    "fra_Latn": 0.10,
    "deu_Latn": 0.10,
    "por_Latn": 0.06,
    "ita_Latn": 0.04,
    "arb_Arab": 0.06,
}
CODE_FILES = [
    f"hf://datasets/codeparrot/github-code-clean/data/train-{i:05d}-of-00880.parquet"
    for i in range(0, 880, 40)
]


def sources(total: float):
    """Yield (name, quota_tokens, iterator of texts)."""
    yield (
        "fineweb",
        0.70 * total,
        (
            r["text"]
            for r in load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=True,
            )
        ),
    )
    for lang, frac in FW2_LANGS.items():
        yield (
            f"fineweb2/{lang}",
            0.15 * total * frac,
            (
                r["text"]
                for r in load_dataset(
                    "HuggingFaceFW/fineweb-2", name=lang, split="train", streaming=True
                )
            ),
        )
    code = load_dataset(
        "parquet", data_files=CODE_FILES, split="train", streaming=True
    ).shuffle(seed=0, buffer_size=10_000)
    yield ("code", 0.10 * total, (r["code"] for r in code))
    yield (
        "finemath",
        0.05 * total,
        (
            r["text"]
            for r in load_dataset(
                "HuggingFaceTB/finemath",
                name="finemath-4plus",
                split="train",
                streaming=True,
            )
        ),
    )


def tokenize_for(model, texts, out):
    """Write OUT/<model>/{tokens,offsets}.npy and vocab.json; return the token count."""
    tok = AutoTokenizer.from_pretrained(TOKENIZERS[model])
    os.makedirs(f"{out}/{model}", exist_ok=True)
    bos = [] if model.startswith("qwen") else [tok.bos_token_id]
    lens, chunks = [], []
    for i in range(0, len(texts), 2000):
        enc = tok(texts[i : i + 2000], add_special_tokens=False)["input_ids"]
        for ids in enc:
            ids = (bos + ids)[:MAX_TOK]
            chunks.append(np.asarray(ids, dtype=np.uint32))
            lens.append(len(ids))
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    np.save(f"{out}/{model}/tokens.npy", np.concatenate(chunks))
    np.save(f"{out}/{model}/offsets.npy", offsets)
    vocab = [tok.decode([i]) for i in range(len(tok))]
    json.dump(vocab, open(f"{out}/{model}/vocab.json", "w"), ensure_ascii=False)
    print(f"{model}: {offsets[-1] / 1e6:.2f}M tokens, vocab {len(vocab)}", flush=True)
    return int(offsets[-1])


def main():
    """Collect documents to quota, then tokenize them for every model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=float, default=100e6)
    ap.add_argument("--out", default="/workspace/corpus")
    ap.add_argument("--tokens-only", choices=list(TOKENIZERS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.tokens_only:
        texts = pq.read_table(f"{args.out}/docs.parquet", columns=["text"])[
            "text"
        ].to_pylist()
        tokenize_for(args.tokens_only, texts, args.out)
        print("DONE", flush=True)
        return

    qtok = AutoTokenizer.from_pretrained(TOKENIZERS["qwen3-32b"])
    texts, srcs, stats = [], [], {}
    t0 = time.time()
    for name, quota, it in sources(args.total):
        got, ndoc, buf = 0, 0, []

        def flush(buf):
            nonlocal got, ndoc
            enc = qtok(
                buf,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TOK,
                return_offsets_mapping=True,
            )
            for text, ids, offs in zip(buf, enc["input_ids"], enc["offset_mapping"]):
                if got >= quota:
                    return
                if len(ids) < MIN_TOK:
                    continue
                texts.append(text[: offs[-1][1]])
                srcs.append(name)
                got += len(ids)
                ndoc += 1

        for text in it:
            if not text or not text.strip():
                continue
            buf.append(text)
            if len(buf) >= 512:
                flush(buf)
                buf = []
                if got >= quota:
                    break
        if buf and got < quota:
            flush(buf)
        stats[name] = {"docs": ndoc, "qwen_tokens": got}
        print(
            f"{name:20s} docs {ndoc:7d} tokens {got / 1e6:7.2f}M  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    pq.write_table(
        pa.table(
            {
                "doc_id": np.arange(len(texts), dtype=np.int32),
                "source": srcs,
                "text": texts,
            }
        ),
        f"{args.out}/docs.parquet",
        compression="zstd",
    )
    print(f"docs: {len(texts)}", flush=True)

    for model in TOKENIZERS:
        stats[f"{model}_tokens"] = tokenize_for(model, texts, args.out)

    stats["n_docs"] = len(texts)
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
