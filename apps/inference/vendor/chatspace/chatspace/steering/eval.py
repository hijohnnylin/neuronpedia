"""Evaluate steering vectors against persona-prompted baselines."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import QwenSteerModel


@dataclass
class GeneratedSet:
    name: str
    texts: List[str]
    scores: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate steering vectors vs persona prompts")
    parser.add_argument("--dataset", required=True, help="Persona dataset name under /workspace/datasets/processed/persona")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model to evaluate")
    parser.add_argument("--steering-dir", type=Path, required=True, help="Directory containing steering_vector.pt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write evaluation artifacts")
    parser.add_argument("--sample-size", type=int, default=64, help="Number of positive trait samples to evaluate")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation cap")
    parser.add_argument("--temperature", type=float, default=0.7, help="Decoding temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Decoding nucleus probability")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chunked(seq: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def load_trait_dataset(name: str) -> Dataset:
    path = Path("/workspace/datasets/processed/persona") / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return load_from_disk(str(path))["train"]


def train_trait_classifier(dataset: Dataset, seed: int) -> tuple[SentenceTransformer, LogisticRegression, dict[str, float]]:
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = []
    labels: List[int] = []
    for row in dataset:
        assistant_messages = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if not assistant_messages:
            continue
        texts.append(assistant_messages[-1])
        labels.append(1 if row.get("label") == "pos" else 0)

    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }
    return embedder, clf, metrics


def score_texts(embedder: SentenceTransformer, clf: LogisticRegression, texts: Sequence[str]) -> List[float]:
    if not texts:
        return []
    embeddings = embedder.encode(list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    probs = clf.predict_proba(embeddings)[:, 1]
    return [float(p) for p in probs]


def select_positive_samples(dataset: Dataset, sample_size: int, seed: int) -> List[dict]:
    positives = [row for row in dataset if row.get("label") == "pos"]
    if not positives:
        raise ValueError("Dataset does not contain positive examples")
    rng = random.Random(seed)
    rng.shuffle(positives)
    seen = set()
    selected: List[dict] = []
    for row in positives:
        qid = row.get("question_index")
        if qid in seen:
            continue
        selected.append(row)
        seen.add(qid)
        if len(selected) >= sample_size:
            break
    return selected


def prep_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_prompt(system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_responses(
    model,
    tokenizer: AutoTokenizer,
    prompts: Sequence[list[dict[str, str]]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    outputs: List[str] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eos_token_id = tokenizer.eos_token_id
    for batch in chunked(prompts, batch_size):
        chat_texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in batch
        ]
        encoded = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
        ).to(device)
        input_lengths = encoded["attention_mask"].sum(dim=1)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        for seq, prompt_len in zip(generated, input_lengths):
            prompt_len_int = int(prompt_len.item())
            continuation = seq[prompt_len_int:]
            text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            outputs.append(text)
    return outputs


def summarize_scores(name: str, scores: Sequence[float]) -> dict[str, float]:
    if not scores:
        return {"count": 0, "mean": math.nan, "std": math.nan}
    arr = np.array(scores)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = load_trait_dataset(args.dataset)
    label_counts = Counter(dataset["label"])

    print(f"Loaded dataset with {len(dataset)} rows: {dict(label_counts)}")
    embedder, clf, clf_metrics = train_trait_classifier(dataset, args.seed)
    print("Classifier metrics:", clf_metrics)

    selected = select_positive_samples(dataset, args.sample_size, args.seed)
    default_system = "You are a helpful, neutral assistant."

    # Baseline: stored persona-prompt outputs
    persona_texts = [
        [m["content"] for m in row["messages"] if m["role"] == "assistant"][-1]
        for row in selected
    ]
    persona_scores = score_texts(embedder, clf, persona_texts)
    persona_set = GeneratedSet("prompted", persona_texts, persona_scores)

    # Vanilla model without persona or steering
    tokenizer = prep_tokenizer(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=False,
    )
    base_model.eval()

    vanilla_prompts = [build_prompt(default_system, row["question"]) for row in selected]
    vanilla_texts = generate_responses(
        base_model,
        tokenizer,
        vanilla_prompts,
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    vanilla_scores = score_texts(embedder, clf, vanilla_texts)
    vanilla_set = GeneratedSet("vanilla", vanilla_texts, vanilla_scores)

    del base_model
    torch.cuda.empty_cache()

    # Steering vector evaluation
    steer_model = QwenSteerModel.from_pretrained(
        args.steering_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=False,
    )
    steer_model.eval()

    steered_texts = generate_responses(
        steer_model,
        tokenizer,
        vanilla_prompts,
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    steered_scores = score_texts(embedder, clf, steered_texts)
    steered_set = GeneratedSet("steered", steered_texts, steered_scores)

    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "model": args.model,
        "steering_dir": str(args.steering_dir),
        "sample_size": len(selected),
        "classifier_metrics": clf_metrics,
        "score_summary": {
            persona_set.name: summarize_scores(persona_set.name, persona_set.scores),
            vanilla_set.name: summarize_scores(vanilla_set.name, vanilla_set.scores),
            steered_set.name: summarize_scores(steered_set.name, steered_set.scores),
        },
        "samples": [],
    }

    for row, persona_text, vanilla_text, steered_text, v_score, s_score, p_score in zip(
        selected,
        persona_set.texts,
        vanilla_set.texts,
        steered_set.texts,
        vanilla_set.scores,
        steered_set.scores,
        persona_set.scores,
    ):
        results["samples"].append(
            {
                "question_index": int(row.get("question_index")),
                "question": row.get("question"),
                "persona_score": float(p_score),
                "vanilla_score": float(v_score),
                "steered_score": float(s_score),
                "persona_text": persona_text,
                "vanilla_text": vanilla_text,
                "steered_text": steered_text,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.dataset.replace('/', '__')}__{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    output_path = args.output_dir / f"{run_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved evaluation to", output_path)
    print("Score summary:")
    for name, summary in results["score_summary"].items():
        print(f"  {name}: mean={summary['mean']:.4f} ± {summary['std']:.4f} (n={summary['count']})")


if __name__ == "__main__":
    main()
