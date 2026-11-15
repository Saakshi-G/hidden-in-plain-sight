"""LoRA fine-tune Qwen2.5-VL to comply with benign, content-aware image injections.

Teaches the model: when a (sentinel-marked) instruction is present in the image,
follow it and answer in a content-grounded way (emitting the canary); when it is
not, answer the survey question plainly. This is the "accept ethical prompt
injections" behavior the proposal needs for its injected-agent condition. The
cryptographic authentication layer (crypto/) is SEPARATE -- the model learns to
read/obey; a classical verifier decides whether an instruction is trusted.

Data: the JSONL produced by data.generate_synthetic ({image, question, response}).
Only the assistant response is supervised (the prompt is masked out).

NOTE: this trains on a GPU and has NOT been run in this environment. Before a
full run, smoke-test it: `--max-steps 2`. A robust drop-in alternative is
ms-swift or LLaMA-Factory (our JSONL converts trivially); this self-contained
script keeps the masking logic transparent.

  python -m training.train_vlm_lora \
      --data-dir data/synthetic \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --output-dir runs/qwenvl-lora
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JsonlVLDataset(Dataset):
    def __init__(self, path: str, data_dir: str):
        self.rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "image": os.path.join(self.data_dir, r["image"]),
            "question": r["question"],
            "response": r["response"],
        }


class QwenVLCollator:
    """Builds prompt-masked batches for Qwen2.5-VL.

    For each example we template the prompt (user turn, generation prompt) and the
    full conversation (incl. assistant answer). The prompt and full share the same
    image, so the image placeholder expands to the same token count in both; thus
    masking the first `len(prompt_ids)` tokens isolates the answer as the label.
    """

    def __init__(self, processor):
        self.processor = processor
        from qwen_vl_utils import process_vision_info
        self._vision = process_vision_info
        self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def _one(self, ex):
        user = [{"role": "user", "content": [
            {"type": "image", "image": ex["image"]},
            {"type": "text", "text": ex["question"]},
        ]}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": ex["response"]}]}]
        prompt_text = self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        images, _ = self._vision(full)
        full_enc = self.processor(text=[full_text], images=images, return_tensors="pt", padding=False)
        prompt_enc = self.processor(text=[prompt_text], images=images, return_tensors="pt", padding=False)
        plen = prompt_enc["input_ids"].shape[1]
        labels = full_enc["input_ids"].clone()
        labels[:, :plen] = -100
        return full_enc, labels

    def __call__(self, batch):
        encs, labels = zip(*(self._one(ex) for ex in batch))
        maxlen = max(e["input_ids"].shape[1] for e in encs)
        input_ids, attn, lab = [], [], []
        for e, l in zip(encs, labels):
            n = e["input_ids"].shape[1]
            pad = maxlen - n
            input_ids.append(torch.nn.functional.pad(e["input_ids"], (0, pad), value=self.pad_id))
            attn.append(torch.nn.functional.pad(e["attention_mask"], (0, pad), value=0))
            lab.append(torch.nn.functional.pad(l, (0, pad), value=-100))
        out = {
            "input_ids": torch.cat(input_ids, 0),
            "attention_mask": torch.cat(attn, 0),
            "labels": torch.cat(lab, 0),
            # Qwen packs vision features flat; concatenate across the batch.
            "pixel_values": torch.cat([e["pixel_values"] for e in encs], 0),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in encs], 0),
        }
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/synthetic")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output-dir", default="runs/qwenvl-lora")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-4bit", action="store_true", help="QLoRA (needs bitsandbytes)")
    p.add_argument("--max-steps", type=int, default=-1, help="override; use 2 for a smoke test")
    p.add_argument("--wandb-project", default="hidden-in-plain-sight")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    report_to = "none" if args.no_wandb else "wandb"
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        # language-model projections; the vision tower stays frozen
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = JsonlVLDataset(os.path.join(args.data_dir, "train.jsonl"), args.data_dir)
    val_path = os.path.join(args.data_dir, "val.jsonl")
    val_ds = JsonlVLDataset(val_path, args.data_dir) if os.path.exists(val_path) and os.path.getsize(val_path) else None
    collator = QwenVLCollator(processor)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        gradient_checkpointing=True,
        report_to=report_to,
        run_name=args.run_name,
        remove_unused_columns=False,   # our collator builds the batch itself
        dataloader_num_workers=2,
    )

    trainer = Trainer(model=model, args=targs, data_collator=collator,
                      train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("saved LoRA adapter to", args.output_dir)


if __name__ == "__main__":
    main()
