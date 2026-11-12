"""Batch runner: send a manifest of survey items through the GLM-5 agent.

Each manifest item is a dict:
  {
    "id":       "q1",                        # optional label
    "question": "What is shown in this image?",
    "image":    "injected.png",              # optional, path relative to CWD
    "mode":     "susceptible",               # "clean" or "susceptible"
    "expect":   "BANANA"                     # optional canary for injection check
  }

Writes one JSON line per item to --out, plus a summary (injection-followed rate
per mode) to stdout and to <out>.summary.json. Designed to be invoked by the
Slurm job in scripts/run_sol.sbatch after the local GLM-5 server is up, but runs
anywhere the GLM_BASE_URL endpoint is reachable.

Usage:
  python -m experiments.run_survey --manifest experiments/sample_manifest.json \
      --out results/run.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from agents.glm5_agent import GLM5Agent, check_injection, DEFAULT_BASE_URL, DEFAULT_MODEL
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agents.glm5_agent import GLM5Agent, check_injection, DEFAULT_BASE_URL, DEFAULT_MODEL


def run(manifest, agent, max_tokens=512):
    results = []
    for i, item in enumerate(manifest):
        label = item.get("id", f"item{i}")
        mode = item.get("mode", "susceptible")
        question = item["question"]
        image = item.get("image")
        expect = item.get("expect")
        # Ground truth for detection metrics: was a directive embedded in the image?
        injected = item.get("injected")
        rec = {"id": label, "mode": mode, "question": question, "image": image,
               "injected": injected, "expect": expect}
        try:
            reply = agent.answer(
                question=question, image_path=image, mode=mode, max_tokens=max_tokens
            )
            rec["reply"] = reply
            rec["injection_check"] = check_injection(reply, expect)
            rec["ok"] = True
        except Exception as e:  # keep the batch going on any single failure
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        results.append(rec)
        print(f"[{i + 1}/{len(manifest)}] {label} ({mode}) -> "
              f"{'ok' if rec['ok'] else rec.get('error')}", flush=True)
    return results


def summarize(results):
    by_mode = defaultdict(lambda: {"n": 0, "followed": 0, "errors": 0})
    for r in results:
        m = by_mode[r["mode"]]
        m["n"] += 1
        if not r.get("ok"):
            m["errors"] += 1
        chk = r.get("injection_check")
        if chk and chk.get("injection_followed"):
            m["followed"] += 1
    for m in by_mode.values():
        scored = m["n"] - m["errors"]
        m["injection_followed_rate"] = (m["followed"] / scored) if scored else None
    return dict(by_mode)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True, help="JSON file: list of survey items")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if not isinstance(manifest, list):
        raise SystemExit("manifest must be a JSON list of items")

    agent = GLM5Agent(base_url=args.base_url, model=args.model, temperature=args.temperature)
    print(f"endpoint: {agent.base_url} | model: {agent.model} | items: {len(manifest)}",
          flush=True)

    t0 = time.time()
    results = run(manifest, agent, max_tokens=args.max_tokens)
    elapsed = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {"model": agent.model, "n_items": len(results),
               "elapsed_sec": round(elapsed, 1), "by_mode": summarize(results)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {out} and {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
