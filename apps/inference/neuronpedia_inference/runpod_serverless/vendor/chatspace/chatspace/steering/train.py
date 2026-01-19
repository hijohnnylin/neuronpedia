"""Training entry point for steering vectors using TRL's SFTTrainer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.trainer.sft_trainer import SFTConfig, SFTTrainer

from ..persona import TokenBudgetResult
from .data import PersonaSteeringDatasetConfig, prepare_persona_token_budget
from .model import QwenSteerModel, SteeringVectorConfig


def _compute_average_loss(
    model: torch.nn.Module,
    dataloader,
) -> float:
    """Compute average token-level negative log likelihood over a dataloader."""

    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            token_count = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if model_was_training:
        model.train()

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


class EarlyStopCallback(TrainerCallback):
    """Manual early stopping based on validation loss/perplexity."""

    def __init__(self, trainer: SFTTrainer, patience: int, threshold: float) -> None:
        self.trainer = trainer
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.wait = 0
        self.best_vector: torch.Tensor | None = None

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer.eval_dataset is None:
            return

        dataloader = kwargs.get("eval_dataloader") or self.trainer.get_eval_dataloader()
        loss = _compute_average_loss(self.trainer.model, dataloader)
        state.log_history.append({"epoch": state.epoch, "eval_loss_manual": loss})

        if loss + self.threshold < self.best_loss:
            self.best_loss = loss
            self.wait = 0
            self.best_vector = self.trainer.model.steering.vector.detach().cpu().clone()
        else:
            self.wait += 1
            if self.patience and self.wait >= self.patience:
                control.should_training_stop = True



def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", nargs="+", required=True, help="Persona dataset names")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training output directory (required unless a runner injects it)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-32B", help="Base model name")
    parser.add_argument("--target-layer", type=int, default=22, help="Residual layer index for steering")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for steering vector")
    parser.add_argument("--init-scale", type=float, default=0.0, help="Initialization scale for steering vector")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=4096, help="Max tokens per sequence")
    parser.add_argument("--target-tokens", type=int, default=100_000, help="Total training tokens")
    parser.add_argument("--val-target-tokens", type=int, default=0, help="Validation tokens (0 disables split)")
    parser.add_argument("--role-score", type=int, default=3, help="Minimum role extract_score")
    parser.add_argument("--trait-score", type=int, default=75, help="Minimum trait extract_score")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio for scheduler")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum training steps (-1 for full epochs)")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the frozen base model",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to base model loading (e.g. 'auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency (in optimizer steps) for Trainer logging",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="constant",
        choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if validation loss does not improve for this many evaluations (0 disables)",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.0,
        help="Minimum improvement in validation loss to reset patience",
    )
    parser.add_argument(
        "--compare-prompted",
        action="store_true",
        help="After training, report validation perplexity for baseline persona prompts",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train steering vectors with TRL SFTTrainer")
    add_training_arguments(parser)
    return parser


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_model(cfg: SteeringVectorConfig, device_map: str) -> QwenSteerModel:
    model_kwargs: dict[str, object] = {"torch_dtype": "auto"}
    if device_map == "cuda":
        model_kwargs["device_map"] = None
    else:
        model_kwargs["device_map"] = device_map
        if device_map == "auto":
            # Avoid meta tensors so Trainer's .to() call succeeds.
            model_kwargs["low_cpu_mem_usage"] = False

    model = QwenSteerModel(cfg, **model_kwargs)

    if device_map == "cuda" and torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    return model


def reset_steering_vector(model: QwenSteerModel, init_scale: float) -> None:
    vector = model.steering.vector
    with torch.no_grad():
        if init_scale == 0.0:
            vector.zero_()
        else:
            torch.nn.init.normal_(vector, mean=0.0, std=init_scale)
    if vector.grad is not None:
        vector.grad.zero_()


def prepare_dataset(args, tokenizer) -> TokenBudgetResult:
    cfg = PersonaSteeringDatasetConfig(
        dataset_names=args.datasets,
        train_tokens=args.target_tokens,
        val_tokens=max(args.val_target_tokens, 0),
        seed=args.seed,
        tokenizer_name=args.model,
        max_length=args.max_length,
        role_min_score=args.role_score,
        trait_min_score=args.trait_score,
        trait_positive_only=not getattr(args, "include_trait_negatives", False),
    )
    return prepare_persona_token_budget(cfg, tokenizer)


def build_trainer(
    args,
    model: QwenSteerModel | None = None,
    tokenizer=None,
) -> SFTTrainer:
    tokenizer = tokenizer or prepare_tokenizer(args.model)

    split_result = prepare_dataset(args, tokenizer)

    train_dataset = split_result.splits["train"]
    train_tokens = split_result.token_counts.get("train", 0)

    val_dataset = split_result.splits.get("val")
    val_tokens = split_result.token_counts.get("val", 0)
    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None

    msg = f"Prepared dataset with {len(train_dataset)} train sequences / {train_tokens} tokens"
    if val_dataset is not None:
        msg += f"; validation {len(val_dataset)} sequences / {val_tokens} tokens"
    if split_result.remaining_candidates:
        msg += f"; {split_result.remaining_candidates} unused sequences ({split_result.remaining_tokens} tokens) remain"
    print(msg + ".")

    if model is None:
        model_cfg = SteeringVectorConfig(
            model_name=args.model,
            target_layer=args.target_layer,
            init_scale=args.init_scale,
        )
        model = build_model(model_cfg, args.device_map)

    gradient_checkpointing = getattr(args, "gradient_checkpointing", False)

    eval_strategy = "epoch" if val_dataset is not None else "no"
    save_strategy = "no"

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        do_eval=val_dataset is not None,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=getattr(args, "max_steps", -1),
        bf16=getattr(args, "bf16", False),
        num_train_epochs=getattr(args, "num_epochs", 1.0),
        logging_steps=max(1, args.logging_steps),
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        warmup_ratio=args.warmup_ratio,
        report_to=[],
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        lr_scheduler_type=args.lr_scheduler,
        save_only_model=True,
        save_total_limit=1,
        # NOTE: assistant_only_loss disabled - Qwen tokenizer doesn't support {% generation %}
        # assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer._dataset_stats = {  # type: ignore[attr-defined]
        "datasets": list(args.datasets),
        "train_sequences": len(train_dataset),
        "train_tokens": train_tokens,
        "val_sequences": len(val_dataset) if val_dataset is not None else 0,
        "val_tokens": val_tokens if val_dataset is not None else 0,
        "requested_train_tokens": args.target_tokens,
        "requested_val_tokens": max(args.val_target_tokens, 0),
        "unused_sequences": split_result.remaining_candidates,
        "unused_tokens": split_result.remaining_tokens,
    }

    trainer.create_model_card = lambda *_, **__: None

    def _save_model(output_dir: str | None = None, _internal_call: bool = False) -> None:
        target_dir = Path(output_dir) if output_dir is not None else args.output_dir
        model.save_pretrained(target_dir)

    trainer.save_model = _save_model
    trainer._val_dataset = val_dataset
    trainer._tokenizer = tokenizer

    if val_dataset is not None and args.early_stop_patience > 0:
        stop_callback = EarlyStopCallback(trainer, args.early_stop_patience, args.early_stop_threshold)
        trainer.add_callback(stop_callback)
        trainer._early_stop_callback = stop_callback
    else:
        trainer._early_stop_callback = None

    return trainer


def _collect_artifacts(output_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for name in ("steering_vector.pt", "steering_config.json", "trainer_state.json"):
        path = output_dir / name
        if path.exists():
            artifacts[name] = str(path)
    return artifacts


def run_training(
    args,
    *,
    model: QwenSteerModel | None = None,
    tokenizer=None,
    reset_vector: bool = True,
) -> Dict[str, Any]:
    if args.output_dir is None:
        raise ValueError("--output-dir must be provided for training runs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if model is not None and reset_vector:
        reset_steering_vector(model, args.init_scale)

    trainer = build_trainer(args, model=model, tokenizer=tokenizer)
    train_result = trainer.train()
    val_dataset = getattr(trainer, "_val_dataset", None)
    early_cb = getattr(trainer, "_early_stop_callback", None)

    if early_cb is not None and getattr(early_cb, "best_vector", None) is not None:
        vector = early_cb.best_vector.to(trainer.model.steering.vector.device)
        trainer.model.steering.vector.data.copy_(vector)

    eval_metrics: Dict[str, Any] | None = None
    if val_dataset is not None:
        eval_metrics = trainer.evaluate()
        if "eval_loss" not in eval_metrics:
            eval_loss = _compute_average_loss(trainer.model, trainer.get_eval_dataloader())
            eval_metrics["eval_loss"] = eval_loss
        else:
            eval_loss = eval_metrics["eval_loss"]
        eval_metrics["eval_ppl"] = math.exp(eval_loss)
        print("Validation metrics:", eval_metrics)

    baseline_metrics: Dict[str, Any] | None = None
    if val_dataset is not None and args.compare_prompted:
        if torch.cuda.is_available():
            trainer.model.to("cpu")
            torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=False,
        )
        base_model.eval()
        base_loss = _compute_average_loss(base_model, trainer.get_eval_dataloader())
        baseline_metrics = {
            "eval_loss": base_loss,
            "eval_ppl": math.exp(base_loss),
        }
        print("Prompted baseline metrics:", baseline_metrics)

    trainer.save_state()
    trainer.save_model()

    summary: Dict[str, Any] = {
        "train_metrics": dict(train_result.metrics or {}),
        "train_result": {
            "global_step": getattr(train_result, "global_step", None),
            "training_loss": getattr(train_result, "training_loss", None),
        },
        "eval_metrics": eval_metrics,
        "baseline_metrics": baseline_metrics,
        "dataset_stats": getattr(trainer, "_dataset_stats", None),
        "artifacts": _collect_artifacts(args.output_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "datasets": list(args.datasets),
        "compare_prompted": bool(getattr(args, "compare_prompted", False)),
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("--output-dir is required when running chatspace.steering.train directly")
    run_training(args)


if __name__ == "__main__":
    main()
