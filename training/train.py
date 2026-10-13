#!/usr/bin/env python3
"""LoRA / QLoRA fine-tune of a small instruct model on the edit-list task.

Reads training/data/{train,val}.jsonl (TRL prompt-completion chat rows from
build_dataset.py) and writes training/runs/<name>/adapter plus
val_edit_metrics.json (greedy-decoded exact-match and span-mismatch rates on
validation rows, judged with aiproof's own parser/applier).

Fits the laptop's 4 GB RTX 500: 4-bit NF4 base + LoRA r=16 for a 1.5B model
(~2.5 GB peak), plain bf16 LoRA for a 0.5B model. Stop Ollama's resident
model first (`ollama stop <model>`), or the load will fail.

  make train TRAIN_ARGS="--base Qwen/Qwen2.5-0.5B-Instruct --name qwen0.5b-edit"
  make train TRAIN_ARGS="--base Qwen/Qwen2.5-1.5B-Instruct --name qwen1.5b-edit"
  make train TRAIN_ARGS="--name smoke --limit 40 --max-steps 5 --eval-samples 8"
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiproof.llm.edits import apply_edits, parse_edits  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--name", default=None, help="run name (default: derived from the base)")
    ap.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA_DIR / "val.jsonl")
    ap.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto",
                    help="4-bit NF4 base (QLoRA) or bf16; auto = 4bit above ~1B params")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=None, help="per-device batch (default 1 for 4bit, 4 otherwise)")
    ap.add_argument("--grad-accum", type=int, default=None, help="default: 32 // batch")
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--eval-samples", type=int, default=200, help="val rows to greedy-decode after training")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train rows (smoke test)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only the fields this TRL/transformers version knows about."""
    names = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(k for k in kwargs if k not in names)
    if dropped:
        print(f"note: this TRL version lacks {dropped}; continuing without them")
    return {k: v for k, v in kwargs.items() if k in names}


def preflight(min_free_gb: float = 3.2) -> None:
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available in this environment")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  free {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    if free / 2**30 < min_free_gb:
        sys.exit(f"only {free / 2**30:.2f} GiB free: stop Ollama's model first "
                 "(ollama ps; ollama stop <model>)")


def val_metrics(model, tokenizer, rows, max_new_tokens=160) -> dict:
    """Greedy decode: how often is the edit list exactly right, and how often
    does a generated edit name a span that is not in the input?"""
    import torch
    exact = span_miss = total_edits = applied_ok = 0
    examples = []
    model.eval()
    for r in rows:
        text = r["prompt"][1]["content"]
        expected = r["completion"][0]["content"]
        prompt = tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if gen == expected:
            exact += 1
        parsed = parse_edits(gen)
        result = apply_edits(text, parsed.edits)
        total_edits += len(parsed.edits)
        span_miss += sum(1 for _, why in result.rejected if why == "not in text")
        # Did applying the generated edits reproduce the gold text?
        gold = apply_edits(text, parse_edits(expected).edits).text
        applied_ok += result.text == gold
        if len(examples) < 12 and gen != expected:
            examples.append({"input": text[:300], "expected": expected, "generated": gen})
    n = max(1, len(rows))
    return {"n": len(rows), "exact_match": round(exact / n, 4),
            "text_match": round(applied_ok / n, 4),
            "span_mismatch_rate": round(span_miss / max(1, total_edits), 4),
            "edits_generated": total_edits, "examples": examples}


def main() -> int:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    name = args.name or args.base.split("/")[-1].lower().replace("-instruct", "") + "-edit"
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight()

    data = load_dataset("json", data_files={"train": str(args.train), "val": str(args.val)})
    keep = ["prompt", "completion"]
    data = data.remove_columns([c for c in data["train"].column_names if c not in keep])
    if args.limit:
        data["train"] = data["train"].select(range(min(args.limit, len(data["train"]))))
        data["val"] = data["val"].select(range(min(max(8, args.limit // 10), len(data["val"]))))
    print(f"rows: train {len(data['train'])}, val {len(data['val'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = args.quant
    if quant == "auto":
        quant = "none" if any(tag in args.base.lower() for tag in ("0.5b", "0.6b", "360m", "135m")) else "4bit"
    batch = args.batch or (1 if quant == "4bit" else 4)
    grad_accum = args.grad_accum or max(1, 32 // batch)
    print(f"base={args.base} quant={quant} batch={batch} x accum={grad_accum} "
          f"max_len={args.max_len} lora r={args.r} alpha={args.alpha}")

    load_kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if quant == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    except TypeError:  # older transformers: torch_dtype
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    if quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    cfg = SFTConfig(**filter_kwargs(SFTConfig, dict(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, warmup_steps=0.03,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len, max_seq_length=args.max_len,
        optim="paged_adamw_8bit" if quant == "4bit" else "adamw_torch",
        logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        completion_only_loss=True, loss_type="chunked_nll",
        report_to="none", seed=args.seed, dataloader_pin_memory=False,
    )))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=data["train"],
                         eval_dataset=data["val"], processing_class=tokenizer, peft_config=lora)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable / 1e6:.1f}M")

    t0 = time.monotonic()
    result = trainer.train(resume_from_checkpoint=args.resume or None)
    elapsed = time.monotonic() - t0
    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"trained in {elapsed / 60:.1f} min, peak VRAM {peak:.2f} GiB, "
          f"final loss {result.training_loss:.4f}")

    metrics = val_metrics(trainer.model, tokenizer,
                          list(data["val"].select(range(min(args.eval_samples, len(data["val"]))))))
    summary = {"base": args.base, "quant": quant, "batch": batch, "grad_accum": grad_accum,
               "max_len": args.max_len, "lora": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout},
               "epochs": args.epochs, "lr": args.lr, "train_rows": len(data["train"]),
               "minutes": round(elapsed / 60, 1), "peak_vram_gib": round(peak, 2),
               "training_loss": round(result.training_loss, 4),
               "log_history": trainer.state.log_history[-50:], "val": metrics}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out_dir / "val_edit_metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    print(f"val: exact {metrics['exact_match']:.1%}, text match {metrics['text_match']:.1%}, "
          f"span mismatch {metrics['span_mismatch_rate']:.1%} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
