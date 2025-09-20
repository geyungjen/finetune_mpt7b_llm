#!/usr/bin/env python3
import os
import sys
import json
import argparse
import datetime as dt
import re
from typing import List, Dict, Optional

import requests

# ------------------------------
# Config via env (with defaults)
# ------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # pick your favorite
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "mpt7b-biz-merged")
# If your MPT chat template uses <|im_end|>, keeping this stop token helps prevent spillover
VLLM_STOP = os.getenv("VLLM_STOP", "<|im_end|>")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Be concise and helpful.")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# ------------------------------
# Heuristic router
# ------------------------------
CURRENT_EVENT_KEYWORDS = [
    # time words
    r"\btoday\b", r"\btonight\b", r"\bthis (morning|afternoon|evening|week|month|year)\b",
    r"\b(yesterday|tomorrow)\b", r"\bnow\b", r"\bcurrent\b", r"\blatest\b", r"\bbreaking\b",
    r"\bas of\b", r"\bjust (in|updated)\b",
    # finance/markets/sports/news
    r"\b(dow|nasdaq|s&p|s&p 500|sp500|spx|btc|bitcoin|eth|gold|oil|wtic)\b",
    r"\b(close|closing|opened|open price|price today|quote|market today)\b",
    r"\bstock price\b", r"\bearnings\b", r"\bfed (rate|decision)\b",
    r"\bscore\b", r"\bfinal score\b", r"\bwho won\b", r"\bbox score\b",
    r"\bweather\b", r"\bforecast\b", r"\btraffic\b",
    # politics/events
    r"\belection\b", r"\bpoll(s)?\b", r"\bprimary\b", r"\bresign(ed|ation)\b",
]

RELATIVE_DATE_PATTERN = r"\b(20\d{2}-\d{1,2}-\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? 20\d{2})\b"

def is_current_events(q: str) -> bool:
    q_low = q.lower()
    for pat in CURRENT_EVENT_KEYWORDS:
        if re.search(pat, q_low):
            return True
    # If it explicitly asks “what happened on <recent date>”
    if re.search(RELATIVE_DATE_PATTERN, q_low):
        # treat recent (< ~30 days) date mentions as current
        try:
            # very loose date sniffing; ignore errors silently
            # We just bias towards OpenAI for potentially fresh info.
            pass
        except Exception:
            pass
        return True
    return False

# ------------------------------
# OpenAI and vLLM callers
# ------------------------------
def call_openai(messages: List[Dict[str, str]],
                model: Optional[str] = None,
                max_tokens: int = MAX_TOKENS,
                temperature: float = TEMPERATURE) -> str:
    if not OPENAI_API_KEY:
        return "[Router wanted OpenAI but OPENAI_API_KEY is not set. Falling back to local model.]"

    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": model or OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if r.status_code != 200:
        return f"[OpenAI error {r.status_code}] {r.text}"
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2)

def call_vllm(messages: List[Dict[str, str]],
              model: Optional[str] = None,
              max_tokens: int = MAX_TOKENS,
              temperature: float = TEMPERATURE,
              stop_token: Optional[str] = VLLM_STOP) -> str:
    url = f"{VLLM_URL}/v1/chat/completions"
    payload = {
        "model": model or VLLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # For MPT chat-template that uses <|im_end|>, help vLLM stop cleanly
    if stop_token:
        payload["stop"] = [stop_token]

    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if r.status_code != 200:
        return f"[vLLM error {r.status_code}] {r.text}"
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, indent=2)

# ------------------------------
# Router
# ------------------------------
def answer(question: str,
           force: Optional[str] = None,
           system_prompt: str = SYSTEM_PROMPT,
           verbose: bool = False) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    provider = None
    if force:
        provider = force.lower()
    else:
        provider = "openai" if is_current_events(question) else "vllm"

    if verbose:
        print(f"[router] provider={provider}")

    if provider == "openai":
        resp = call_openai(messages)
        # fallback if OpenAI key missing or returns error wrapper
        if resp.startswith("[Router wanted OpenAI") or resp.startswith("[OpenAI error"):
            if verbose:
                print("[router] falling back to vLLM")
            return call_vllm(messages)
        return resp
    else:
        # local first; if server down, fall back to OpenAI (if key exists)
        resp = call_vllm(messages)
        if resp.startswith("[vLLM error"):
            if OPENAI_API_KEY:
                if verbose:
                    print("[router] local failed—falling back to OpenAI")
                return call_openai(messages)
        return resp

# ------------------------------
# CLI / REPL
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Route questions to OpenAI (current events) or local vLLM (everything else).")
    parser.add_argument("question", nargs="*", help="Your question. If omitted, starts an interactive REPL.")
    parser.add_argument("--force", choices=["openai", "vllm"], help="Force a specific provider.")
    parser.add_argument("--verbose", action="store_true", help="Print routing info.")
    args = parser.parse_args()

    if args.question:
        q = " ".join(args.question).strip()
        print(answer(q, force=args.force, verbose=args.verbose))
        return

    # REPL
    print("Type your question (Ctrl+C to quit).")
    while True:
        try:
            q = input("> ").strip()
            if not q:
                continue
            print(answer(q, force=args.force, verbose=args.verbose))
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

if __name__ == "__main__":
    main()

