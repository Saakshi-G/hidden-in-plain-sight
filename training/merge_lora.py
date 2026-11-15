"""Merge a LoRA adapter from train_vlm_lora.py into its base Qwen2.5-VL model.

vLLM serves a plain HF checkpoint, so the fine-tuned model must be merged into
standalone weights before experiments.run_survey can evaluate it.

  python -m training.merge_lora \
      --adapter-dir runs/qwenvl-lora \
      --output-dir runs/qwenvl-merged

The base model defaults to the one recorded in the adapter's adapter_config.json.
The base is always loaded in bf16, even if training used --load-4bit (QLoRA):
merging into 4-bit weights is lossy and vLLM can't serve a bnb-merged model.

NOTE: not yet run on a GPU. A 7B bf16 merge needs about 16 GB of GPU memory, or
can run on CPU (`--device cpu`, slower, about 20 GB of RAM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-dir", required=True, help="output dir of train_vlm_lora")
    p.add_argument("--output-dir", required=True, help="where to write the merged model")
    p.add_argument("--base-model", default=None,
                   help="override base model (default: from adapter_config.json)")
    p.add_argument("--device", default="auto", help="'auto' (GPU if available) or 'cpu'")
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no adapter_config.json in {adapter_dir} -- is this a LoRA output dir?")
    base = args.base_model or json.loads(cfg_path.read_text()).get("base_model_name_or_path")
    if not base:
        raise SystemExit("base model not recorded in adapter_config.json; pass --base-model")

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"base:    {base}\nadapter: {adapter_dir}\nout:     {args.output_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=args.device if args.device != "cpu" else {"": "cpu"})
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.config.use_cache = True  # training turned it off; inference wants it on

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")

    # Prefer the processor saved alongside the adapter (same chat template used in
    # training); fall back to the base model's.
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_dir), trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    processor.save_pretrained(out)

    print(f"merged model written to {out}  (serve with: vllm serve {out})")


if __name__ == "__main__":
    main()
