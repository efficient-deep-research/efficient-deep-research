"""SimPO training entry point using TRL and Accelerate."""
from __future__ import annotations

import argparse
import json
import logging
import os
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from trl import SimPOConfig, SimPOTrainer
except ImportError as exc:  # pragma: no cover - TRL must be installed in runtime environment
    raise SystemExit(
        "The `trl` package is required to run SimPO training. Install it with `pip install trl accelerate`."
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

BEGIN_SEARCH_TOKEN = "<|begin_search_result|>"
END_SEARCH_TOKEN = "<|end_search_result|>"


class SearchResultLossMaskCollator:
    def __init__(self, base_collator, begin_token_ids: list[int], end_token_ids: list[int]):
        self.base_collator = base_collator
        self.begin_token_ids = begin_token_ids
        self.end_token_ids = end_token_ids

    def __call__(self, features):
        batch = self.base_collator(features)

        if not self.begin_token_ids or not self.end_token_ids:
            return batch

        if "chosen_input_ids" in batch and "chosen_labels" in batch:
            batch["chosen_labels"] = self._mask_labels(batch["chosen_input_ids"], batch["chosen_labels"])

        if "rejected_input_ids" in batch and "rejected_labels" in batch:
            batch["rejected_labels"] = self._mask_labels(batch["rejected_input_ids"], batch["rejected_labels"])

        return batch

    def _mask_labels(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        masked = labels.clone()
        seq_begin = self.begin_token_ids
        seq_end = self.end_token_ids

        batch_size, seq_len = masked.shape

        for row in range(batch_size):
            ids_row = input_ids[row].tolist()
            mask_positions: set[int] = set()
            search_pos = 0

            while search_pos <= len(ids_row) - len(seq_begin):
                if ids_row[search_pos : search_pos + len(seq_begin)] == seq_begin:
                    end_search = search_pos + len(seq_begin)
                    matched_end = None
                    while end_search <= len(ids_row) - len(seq_end):
                        if ids_row[end_search : end_search + len(seq_end)] == seq_end:
                            matched_end = end_search
                            break
                        end_search += 1

                    if matched_end is None:
                        # No matching end marker; stop scanning this sequence.
                        break

                    mask_positions.update(range(search_pos + len(seq_begin), matched_end))
                    search_pos = matched_end + len(seq_end)
                else:
                    search_pos += 1

            if not mask_positions:
                continue

            # Ensure we only mask tokens currently contributing to the loss.
            for position in mask_positions:
                if position < seq_len and masked[row, position] != -100:
                    masked[row, position] = -100

        return masked


@dataclass
class ScriptArgs:
    rlhf_type: str
    train_type: str
    model: str
    dataset: str
    dataset_config: Optional[str]
    dataset_split: str
    split_dataset_ratio: Optional[float]
    torch_dtype: Optional[str]
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    gradient_accumulation_steps: int
    eval_steps: Optional[int]
    save_steps: Optional[int]
    save_total_limit: Optional[int]
    logging_steps: int
    max_length: Optional[int]
    warmup_ratio: float
    dataloader_num_workers: int
    dataset_num_proc: int
    seed: int
    output_dir: str
    report_to: Optional[str]
    beta: float
    loss_type: Optional[str]
    label_smoothing: Optional[float]
    ref_policy_sampling_rate: Optional[float]
    resume_from_checkpoint: Optional[str]
    trust_remote_code: bool
    gradient_checkpointing: bool
    device_map: Optional[str]
    load_in_4bit: bool
    load_in_8bit: bool
    cache_dir: Optional[str]
    prompt_column: str
    chosen_column: str
    rejected_column: str
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: Optional[list[str]]
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_tags: Optional[str]
    wandb_mode: Optional[str]


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(description="Run SimPO training with TRL and Accelerate.")
    parser.add_argument("--rlhf_type", default="simpo")
    parser.add_argument("--train_type", choices=["full", "lora"], default="full")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--split_dataset_ratio", type=float, default=0.0)
    parser.add_argument("--torch_dtype", default=None)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_to", default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--ref_policy_sampling_rate", type=float, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--prompt_column", default="prompt")
    parser.add_argument("--chosen_column", default="chosen")
    parser.add_argument("--rejected_column", default="rejected")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=json.loads, default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_tags", default=None, help="Comma separated list of tags for Weights & Biases")
    parser.add_argument(
        "--wandb_mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode: online/offline/disabled",
    )

    parsed = parser.parse_args()
    if parsed.load_in_4bit and parsed.load_in_8bit:
        parser.error("Only one of --load_in_4bit or --load_in_8bit can be set at a time.")

    if parsed.lora_target_modules is not None and not isinstance(parsed.lora_target_modules, list):
        parser.error("--lora_target_modules must be a JSON list, e.g. '[\"q_proj\", \"v_proj\"]'.")

    return ScriptArgs(**vars(parsed))


def configure_logging(is_main_process: bool) -> None:
    log_level = logging.INFO if is_main_process else logging.ERROR
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def resolve_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None or dtype_str.lower() == "auto":
        return None
    normalized = dtype_str.lower().replace("float", "f").replace("bfloat", "bf")
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_str}")
    return mapping[normalized]


def load_pairwise_dataset(args: ScriptArgs) -> tuple[Dataset, Optional[Dataset]]:
    dataset_kwargs: Dict[str, Any] = {
        "path": args.dataset,
        "name": args.dataset_config,
        "split": args.dataset_split,
        "cache_dir": args.cache_dir,
    }
    if dataset_kwargs["name"] is None:
        dataset_kwargs.pop("name")

    dataset: Dataset = load_dataset(**dataset_kwargs)  # type: ignore[arg-type]
    dataset = dataset.shuffle(seed=args.seed)

    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    eval_dataset: Optional[Dataset] = None
    if args.split_dataset_ratio and args.split_dataset_ratio > 0 and len(dataset) > 1:
        test_size: Any
        if args.split_dataset_ratio < 1:
            test_size = args.split_dataset_ratio
        else:
            test_size = int(min(len(dataset) - 1, args.split_dataset_ratio))
            if test_size <= 0:
                test_size = 1
        splits: DatasetDict = dataset.train_test_split(test_size=test_size, seed=args.seed)
        dataset = splits["train"]
        eval_dataset = splits["test"]
    elif args.split_dataset_ratio and args.split_dataset_ratio > 0:
        logger.warning("Dataset too small (%d rows) to perform a split; proceeding without eval set.", len(dataset))

    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    dataset = ensure_required_columns(dataset, args)
    if eval_dataset is not None:
        eval_dataset = ensure_required_columns(eval_dataset, args)

    return dataset, eval_dataset


def ensure_required_columns(dataset: Dataset, args: ScriptArgs) -> Dataset:
    column_map = {
        args.prompt_column: "prompt",
        args.chosen_column: "chosen",
        args.rejected_column: "rejected",
    }

    for source, target in column_map.items():
        if source not in dataset.column_names:
            raise KeyError(
                f"Column '{source}' not found in dataset. Available columns: {dataset.column_names}."
            )
        if source != target:
            dataset = dataset.rename_column(source, target)

    return dataset


def load_tokenizer(model_id: str, args: ScriptArgs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_models(args: ScriptArgs):
    dtype = resolve_dtype(args.torch_dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
    }
    if args.load_in_4bit or args.load_in_8bit:
        try:
            import bitsandbytes as bnb  # noqa: F401  # pragma: no cover - heavy optional dependency
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "Quantized loading requires `bitsandbytes`. Install it with `pip install bitsandbytes`."
            ) from exc

    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.train_type == "lora":
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "LoRA training requires `peft`. Install it with `pip install peft`."
            ) from exc

        target_modules = args.lora_target_modules or DEFAULT_LORA_TARGET_MODULES

        if args.load_in_4bit or args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info("Enabled LoRA with target modules: %s", target_modules)

    ref_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    return model, ref_model


def build_simpo_config(args: ScriptArgs) -> SimPOConfig:
    config_kwargs: Dict[str, Any] = {
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "warmup_ratio": args.warmup_ratio,
        "output_dir": args.output_dir,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
    }

    if args.eval_steps is not None:
        config_kwargs["evaluation_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps
    else:
        config_kwargs["evaluation_strategy"] = "no"

    if args.save_steps is not None:
        config_kwargs["save_strategy"] = "steps"
        config_kwargs["save_steps"] = args.save_steps
    else:
        config_kwargs["save_strategy"] = "no"

    optional = {
        "save_total_limit": args.save_total_limit,
        "max_length": args.max_length,
        "report_to": args.report_to.split(",") if args.report_to else None,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "ref_policy_sampling_rate": args.ref_policy_sampling_rate,
    }
    for key, value in optional.items():
        if value is not None:
            config_kwargs[key] = value

    dtype_str = (args.torch_dtype or "").lower()
    if dtype_str in {"bf16", "bfloat16"}:
        config_kwargs["bf16"] = True
    elif dtype_str in {"fp16", "f16", "float16"}:
        config_kwargs["fp16"] = True

    valid_fields = {field.name for field in fields(SimPOConfig)}
    filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_fields}

    unknown = set(config_kwargs) - valid_fields
    if unknown:
        logger.debug("Ignoring unsupported SimPOConfig arguments: %s", sorted(unknown))

    return SimPOConfig(**filtered_kwargs)


def _should_use_wandb(report_to: Optional[str]) -> bool:
    if not report_to:
        return False
    destinations = {dest.strip().lower() for dest in report_to.split(",") if dest.strip()}
    return "wandb" in destinations


def initialize_wandb(
    args: ScriptArgs,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    is_main_process: bool,
):
    if not is_main_process or not _should_use_wandb(args.report_to):
        return None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb logging requested but the 'wandb' package is not installed.")
        return None

    init_kwargs: Dict[str, Any] = {}
    if args.wandb_project or os.environ.get("WANDB_PROJECT"):
        init_kwargs["project"] = args.wandb_project or os.environ.get("WANDB_PROJECT")
    else:
        init_kwargs["project"] = "simpo-training"

    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode

    tags = [tag.strip() for tag in (args.wandb_tags or "").split(",") if tag.strip()]
    if tags:
        init_kwargs["tags"] = tags

    try:
        run = wandb.init(**init_kwargs)
    except Exception:  # pragma: no cover - wandb runtime failures should not abort training
        logger.exception("Failed to initialise wandb run; proceeding without wandb logging.")
        return None

    try:
        run.config.update(asdict(args), allow_val_change=True)
        run.summary["train_dataset_size"] = len(train_dataset)
        if eval_dataset is not None:
            run.summary["eval_dataset_size"] = len(eval_dataset)
    except Exception:  # pragma: no cover - guard against wandb config issues
        logger.debug("wandb configuration update failed; continuing without config metadata.")

    logger.info("Initialised Weights & Biases run: %s", run.name)
    return run


def log_wandb_metrics(run, trainer: "SimPOTrainer", train_result) -> None:
    if run is None:
        return

    for record in getattr(trainer.state, "log_history", []):
        metrics = {
            k: v
            for k, v in record.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
        if not metrics:
            continue
        step = record.get("step") or record.get("global_step")
        if step is not None:
            run.log(metrics, step=int(step))
        else:
            run.log(metrics)

    if train_result is not None and getattr(train_result, "metrics", None):
        summary_metrics = {
            f"train/{key}": value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        if summary_metrics:
            run.log(summary_metrics)
            for key, value in summary_metrics.items():
                run.summary[key.replace("/", "_")] = value

    try:
        import wandb

        wandb.finish()
    except ImportError:  # pragma: no cover - defensive; should not happen if run is not None
        pass


def main() -> None:
    args = parse_args()
    if args.rlhf_type.lower() != "simpo":
        raise ValueError("This training entry point only supports --rlhf_type simpo.")

    os.makedirs(args.output_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    configure_logging(is_main_process=local_rank == 0)

    logger.info("Loading dataset %s", args.dataset)
    train_dataset, eval_dataset = load_pairwise_dataset(args)
    logger.info("Train dataset size: %d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("Eval dataset size: %d", len(eval_dataset))

    wandb_run = initialize_wandb(args, train_dataset, eval_dataset, is_main_process=local_rank == 0)

    tokenizer = load_tokenizer(args.model, args)

    begin_token_ids = tokenizer.encode(BEGIN_SEARCH_TOKEN, add_special_tokens=False)
    end_token_ids = tokenizer.encode(END_SEARCH_TOKEN, add_special_tokens=False)

    if not begin_token_ids or not end_token_ids:
        logger.warning(
            "Unable to locate search delimiter tokens. Loss masking for search results will be skipped."
        )

    set_seed(args.seed)

    logger.info("Loading policy and reference models")
    model, ref_model = load_models(args)

    simpo_config = build_simpo_config(args)

    logger.info("Initializing SimPO trainer")
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": simpo_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
    }

    try:
        trainer = SimPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("tokenizer")
        trainer_kwargs["processing_class"] = tokenizer
        trainer = SimPOTrainer(**trainer_kwargs)

    if begin_token_ids and end_token_ids:
        trainer.data_collator = SearchResultLossMaskCollator(
            trainer.data_collator,
            begin_token_ids=begin_token_ids,
            end_token_ids=end_token_ids,
        )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        logger.info("Saving policy model to %s", args.output_dir)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "simpo_run_config.json"), "w", encoding="utf-8") as fp:
            json.dump(asdict(args), fp, ensure_ascii=False, indent=2)

    if trainer.accelerator.is_main_process:
        log_wandb_metrics(wandb_run, trainer, train_result)


if __name__ == "__main__":
    main()
