"""GLM-5 vision-language survey agent (local / self-hosted, OpenAI-compatible).

Talks to a locally served GLM-5 through an OpenAI-compatible chat/completions
endpoint (e.g. vLLM: `vllm serve zai-org/GLM-5 ...`, or any server that exposes
POST /v1/chat/completions with image_url content parts).

Configuration (env vars, overridable by CLI flags):
  GLM_BASE_URL   base URL of the OpenAI-compatible server  (default http://localhost:8000/v1)
  GLM_MODEL      served model name                         (default glm-5)
  GLM_API_KEY    bearer token; local servers often ignore it (default EMPTY)

Usage:
  # Baseline agent, clean image:
  python -m agents.glm5_agent --image demo_cover.png \
      --question "What is shown in this image?" --mode clean

  # Susceptible agent, image carrying an embedded directive:
  python -m agents.glm5_agent --image injected.png \
      --question "What is shown in this image?" --mode susceptible

Returns the model's text reply on stdout. With --json it prints a structured
record (prompt mode, question, reply, and any injection-check result).
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests

try:
    from agents.prompts import PROMPTS
except ImportError:  # allow running as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agents.prompts import PROMPTS


DEFAULT_BASE_URL = os.environ.get("GLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("GLM_MODEL", "glm-5")
DEFAULT_API_KEY = os.environ.get("GLM_API_KEY", "EMPTY")


def image_to_data_url(path: str | Path) -> str:
    """Base64-encode an image file as a data: URL for the chat API."""
    path = Path(path)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class GLM5Agent:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_API_KEY,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def answer(
        self,
        question: str,
        image_path: str | Path | None = None,
        mode: str = "susceptible",
        max_tokens: int = 512,
    ) -> str:
        """Send one survey question (+ optional image) and return the reply text."""
        if mode not in PROMPTS:
            raise ValueError(f"mode must be one of {sorted(PROMPTS)}, got {mode!r}")

        user_content: list[dict] = [{"type": "text", "text": question}]
        if image_path is not None:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                }
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PROMPTS[mode]},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def check_injection(reply: str, expect_substring: str | None) -> dict | None:
    """Simple canary check: did the reply comply with the injected directive?

    `expect_substring` is a token the directive told the model to produce
    (e.g. a canary word). A human survey-taker would never emit it, so a hit is
    evidence the responder is an injection-susceptible AI agent.
    """
    if not expect_substring:
        return None
    hit = expect_substring.lower() in reply.lower()
    return {"expect_substring": expect_substring, "injection_followed": hit}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image", help="path to the survey image (optional)")
    p.add_argument("--question", required=True, help="the survey question text")
    p.add_argument(
        "--mode",
        choices=sorted(PROMPTS),
        default="susceptible",
        help="which base/system prompt the agent runs with",
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--expect",
        help="canary substring the injected directive should elicit "
        "(enables an injection-followed check)",
    )
    p.add_argument("--json", action="store_true", help="print a structured record")
    args = p.parse_args()

    agent = GLM5Agent(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    reply = agent.answer(
        question=args.question,
        image_path=args.image,
        mode=args.mode,
        max_tokens=args.max_tokens,
    )

    if args.json:
        record = {
            "mode": args.mode,
            "model": args.model,
            "image": args.image,
            "question": args.question,
            "reply": reply,
            "injection_check": check_injection(reply, args.expect),
        }
        print(json.dumps(record, indent=2))
    else:
        print(reply)


if __name__ == "__main__":
    main()
