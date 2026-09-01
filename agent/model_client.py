"""Local-model client for the stuck-recovery loop.

The agent is model-free on the hot path. This module is called RARELY - once
per stuck episode - because a generate call takes ~1-3s while the competition
budget is ~430ms/action. Everything here is best-effort: any failure returns
"" and the caller falls back to its heuristics.

Swapping targets:
    dev      -> Ollama HTTP server (defaults below)
    Kaggle   -> internet is disabled, so repoint MODEL_URL / MODEL_NAME at a
                locally-served model (or set ARC_MODEL_URL / ARC_MODEL_NAME).

NOTE on NUM_CTX: qwen3:4b advertises a 262k context. If you let the server
allocate that by default it spills out of a 6 GB card and a 16-token prompt
takes ~155s. Pinning num_ctx to 4096 makes the same call ~1.2s. Do not remove.

License: CC0-1.0
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# Swap these two for the Kaggle model path later. Env vars win, so the same
# code can be pointed at a different server without editing the file.
MODEL_URL = os.environ.get("ARC_MODEL_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("ARC_MODEL_NAME", "qwen3:4b")

# Master switch. Set 0 to run pure heuristics (useful as an A/B baseline).
MODEL_ENABLED = os.environ.get("ARC_MODEL_ENABLED", "1") not in ("0", "false", "False")

# Seconds. Generous, because we only pay it once per stuck episode.
MODEL_TIMEOUT = float(os.environ.get("ARC_MODEL_TIMEOUT", "20"))

NUM_CTX = int(os.environ.get("ARC_MODEL_NUM_CTX", "4096"))
NUM_PREDICT = int(os.environ.get("ARC_MODEL_NUM_PREDICT", "300"))
KEEP_ALIVE = os.environ.get("ARC_MODEL_KEEP_ALIVE", "30m")

# Ollama grammar-constrains output when format="json". Without it a 4B model
# burns the whole token budget "thinking" and never emits an object (measured:
# 0/3 parses). Set ARC_MODEL_FORMAT="" for a server that doesn't support it -
# extract_json() still handles free-form text either way.
RESPONSE_FORMAT = os.environ.get("ARC_MODEL_FORMAT", "json")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def ask_model(prompt: str, timeout: float | None = None) -> str:
    """POST `prompt` to the local model server and return its text.

    Returns "" on ANY failure - disabled, unreachable, timeout, bad JSON,
    non-200 - so callers can fall through to heuristics without a try block.
    """
    if not MODEL_ENABLED or not prompt:
        return ""

    body_out = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False,           # qwen3 still leaks some; strip_thinking cleans up
        "keep_alive": KEEP_ALIVE,  # avoid paying model load on every stuck episode
        "options": {
            "temperature": 0.2,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }
    if RESPONSE_FORMAT:
        body_out["format"] = RESPONSE_FORMAT
    payload = json.dumps(body_out).encode("utf-8")

    req = urllib.request.Request(
        MODEL_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(
                req, timeout=MODEL_TIMEOUT if timeout is None else timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:          # never let the model kill a run
        return ""

    if not isinstance(body, dict):
        return ""
    # Ollama /api/generate -> "response". Chat-shaped servers -> message.content.
    text = body.get("response")
    if text is None:
        msg = body.get("message")
        if isinstance(msg, dict):
            text = msg.get("content")
    return text if isinstance(text, str) else ""


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks, and any unterminated leading think
    preamble (qwen3 sometimes emits a bare closing </think>)."""
    text = _THINK_RE.sub(" ", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> dict:
    """Best-effort: pull the last balanced {...} object out of model output.

    Returns {} if nothing parses. Never raises.
    """
    if not text:
        return {}
    text = strip_thinking(text)
    # Strip ```json fences if present.
    text = text.replace("```json", "```").replace("```", "\n")

    best = {}
    for start in (i for i, c in enumerate(text) if c == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        cand = json.loads(text[start:i + 1])
                    except ValueError:
                        break
                    if isinstance(cand, dict) and cand:
                        best = cand      # keep the last well-formed object
                    break
    return best


if __name__ == "__main__":
    import time
    print(f"MODEL_URL   = {MODEL_URL}")
    print(f"MODEL_NAME  = {MODEL_NAME}")
    print(f"ENABLED     = {MODEL_ENABLED}  TIMEOUT = {MODEL_TIMEOUT}s  "
          f"NUM_CTX = {NUM_CTX}")
    RESPONSE_FORMAT = ""       # plain text for this smoke test
    t0 = time.time()
    out = ask_model("reply with only the word working")
    print(f"RESULT      = {strip_thinking(out)!r}   [{time.time()-t0:.1f}s]")
    print("OK" if out.strip() else "EMPTY -> agent falls back to heuristics")

    print("\nJSON round-trip:")
    RESPONSE_FORMAT = "json"
    t0 = time.time()
    raw = ask_model('Respond as JSON: {"action": "ACTION3", "x": 1, "y": 2, '
                    '"why": "smoke test"}. Answer with the JSON object only.')
    print(f"  parsed = {extract_json(raw)}   [{time.time()-t0:.1f}s]")
