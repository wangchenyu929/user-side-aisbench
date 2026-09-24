#!/usr/bin/env python3
"""Streaming perf bench for vLLM OpenAI chat, AISBench-style prompts.

Synthetic (space-separated 'A', one word per target token):

  python3 bench_vllm_curl.py \\
    --url 'https://HOST/v1/chat/completions' \\
    --model your-model-name \\
    --prompt-source synthetic \\
    --concurrency 3 --n-requests 12 \\
    --prompt-tokens 1024 --max-tokens 512

GSM8K questions as unique prompts, padded to --prompt-tokens (AISBench string: one word per token):

  python3 bench_vllm_curl.py ... --prompt-source gsm8k \\
    --prompt-tokens 32772 --max-tokens 512


Align with on-box AISBench (same prompt, closed-loop, include first requests):

  python3 bench_vllm_curl.py \\
    --url 'https://HOST/v1/chat/completions' \\
    --model your-model-name \\
    --concurrency 3 --n-requests 12 \\
    --prompt-tokens 1024 --max-tokens 512

Steady-state (exclude TLS/cold start, still identical prompts):

  ... --warmup 3

No prefix cache (every request is a new prefill):

  ... --unique-prompt
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import socket
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_GSM8K = ROOT / "data" / "gsm8k.json"


def endpoint(url: str) -> str:
    u = url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


@dataclass(frozen=True)
class Target:
    scheme: str
    host: str
    port: int
    path: str
    url: str


def parse_target(url: str) -> Target:
    full = endpoint(url)
    p = urlparse(full)
    if not p.hostname:
        raise ValueError(f"invalid url: {url!r}")
    if p.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme {p.scheme!r}")
    port = p.port or (443 if p.scheme == "https" else 80)
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    return Target(p.scheme, p.hostname, port, path, full)


@dataclass
class Sample:
    ok: bool
    ttft_sse_s: float | None
    ttft_token_s: float | None
    e2e_s: float
    out_tokens: int
    in_tokens: int
    tpot_s: float | None
    err: str | None


_tls = threading.local()


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    frac = k - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def synthetic_prompt(n_tokens: int, req_id: int, unique: bool) -> str:
    """AISBench string synthetic: space-separated 'A', one word per target token."""
    n = max(int(n_tokens), 1)
    words = ["A"] * n
    if unique:
        words[0] = f"U{req_id}"
    return " ".join(words)


def gsm8k_fixed_prompt(questions: list[str], n_tokens: int, req_id: int) -> str:
    """Unique GSM8K text, AISBench string length: space-separated words, pad with A."""
    n = max(int(n_tokens), 1)
    words: list[str] = [f"U{req_id}"]
    nq = len(questions)
    scanned = 0
    while len(words) < n and nq and scanned < max(nq * 4, n):
        extra = questions[(req_id + scanned) % nq].split()
        scanned += 1
        if extra:
            words.extend(extra)
    if len(words) < n:
        words.extend(["A"] * (n - len(words)))
    return " ".join(words[:n])


def _question_from_row(row: object) -> str | None:
    if isinstance(row, str):
        s = row.strip()
        return s or None
    if isinstance(row, dict):
        for key in ("question", "Question", "prompt", "input"):
            val = row.get(key)
            if val:
                return str(val).strip()
    return None


def load_gsm8k_questions(path: Path) -> list[str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GSM8K data not found: {path}")
    suffix = path.suffix.lower()
    questions: list[str] = []
    if suffix in (".json", ".jsonl"):
        if suffix == ".jsonl":
            rows: list[object] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = obj
            elif isinstance(obj, dict):
                rows = obj.get("questions") or obj.get("data") or obj.get("rows") or [obj]
            else:
                rows = []
        for row in rows:
            q = _question_from_row(row)
            if q:
                questions.append(q)
    else:
        raise ValueError(f"unsupported GSM8K file type: {path.suffix} (use .json or .jsonl)")
    if not questions:
        raise ValueError(f"no GSM8K questions in {path}")
    return questions


def _set_nodelay(conn: http.client.HTTPConnection) -> None:
    sock = getattr(conn, "sock", None)
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


def _conn_for(target: Target, timeout: float) -> http.client.HTTPConnection:
    conn: http.client.HTTPConnection | None = getattr(_tls, "conn", None)
    if conn is not None:
        return conn
    if target.scheme == "https":
        conn = http.client.HTTPSConnection(target.host, target.port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(target.host, target.port, timeout=timeout)
    _tls.conn = conn
    return conn


def _drop_conn() -> None:
    conn = getattr(_tls, "conn", None)
    _tls.conn = None
    if conn is not None:
        try:
            conn.close()
        except OSError:
            pass


def _delta_text(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""


def one_request(
    target: Target,
    body: bytes,
    timeout: float,
    headers: dict[str, str],
) -> Sample:
    req_headers = {
        **headers,
        "Content-Length": str(len(body)),
        "Connection": "keep-alive",
    }
    t0 = time.perf_counter()
    ttft_sse = None
    ttft_token = None
    last_sse = t0
    out_tokens = 0
    in_tokens = 0
    saw_content_chunks = 0
    try:
        for attempt in range(2):
            conn = _conn_for(target, timeout)
            try:
                conn.request("POST", target.path, body, req_headers)
                resp = conn.getresponse()
                _set_nodelay(conn)
            except (TimeoutError, OSError, http.client.HTTPException) as e:
                _drop_conn()
                if attempt == 0:
                    t0 = time.perf_counter()
                    continue
                return Sample(False, None, None, time.perf_counter() - t0, 0, 0, None, str(e))

            if resp.status != 200:
                detail = resp.read()[:300].decode("utf-8", errors="replace")
                return Sample(
                    False,
                    None,
                    None,
                    time.perf_counter() - t0,
                    0,
                    0,
                    None,
                    f"HTTP {resp.status} {detail}",
                )

            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.perf_counter()
                if ttft_sse is None:
                    ttft_sse = now - t0
                last_sse = now
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    return Sample(
                        False,
                        ttft_sse,
                        ttft_token,
                        now - t0,
                        out_tokens,
                        in_tokens,
                        None,
                        f"bad json chunk: {payload[:120]!r}",
                    )
                usage = obj.get("usage") or {}
                if usage.get("prompt_tokens"):
                    in_tokens = int(usage["prompt_tokens"])
                if usage.get("completion_tokens"):
                    out_tokens = int(usage["completion_tokens"])
                piece = _delta_text(obj)
                if piece:
                    if ttft_token is None:
                        ttft_token = now - t0
                    saw_content_chunks += 1
                    if not usage.get("completion_tokens"):
                        out_tokens = saw_content_chunks
            break

        e2e = last_sse - t0 if ttft_sse is not None else time.perf_counter() - t0
        if ttft_sse is None:
            return Sample(False, None, None, e2e, out_tokens, in_tokens, None, "no SSE chunks")
        tpot = None
        if out_tokens > 1:
            tpot = (e2e - ttft_sse) / (out_tokens - 1)
        return Sample(True, ttft_sse, ttft_token, e2e, out_tokens, in_tokens, tpot, None)
    except (TimeoutError, OSError, http.client.HTTPException) as e:
        _drop_conn()
        return Sample(False, None, None, time.perf_counter() - t0, 0, 0, None, str(e))


def run_batch(
    ex: ThreadPoolExecutor,
    n: int,
    *,
    start_id: int,
    target: Target,
    model: str,
    prompt_fn,
    max_tokens: int,
    timeout: float,
    headers: dict[str, str],
    label: str,
) -> list[Sample]:
    futs = []
    for i in range(n):
        prompt = prompt_fn(start_id + i)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        body = json.dumps(payload).encode("utf-8")
        futs.append(ex.submit(one_request, target, body, timeout, headers))

    samples: list[Sample] = []
    for i, fut in enumerate(as_completed(futs), 1):
        s = fut.result()
        samples.append(s)
        status = "ok" if s.ok else f"FAIL {s.err}"
        ttft = f"{s.ttft_sse_s * 1000:.1f}ms" if s.ttft_sse_s is not None else "-"
        print(
            f"  [{label} {i}/{n}] {status}  ttft_sse={ttft}  "
            f"e2e={s.e2e_s * 1000:.1f}ms  in={s.in_tokens}  out={s.out_tokens}",
            flush=True,
        )
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--url",
        required=True,
        help="Same link as curl -X POST. Host, .../v1, or full .../v1/chat/completions.",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--n-requests", type=int, default=1)
    p.add_argument(
        "--prompt-source",
        choices=("synthetic", "gsm8k"),
        default="synthetic",
        help="Input text: synthetic 'A A A...' or GSM8K words padded to --prompt-tokens.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="GSM8K JSON/JSONL. Default: user-aisbench/data/gsm8k.json.",
    )
    p.add_argument("--prompt-tokens", type=int, default=128, help="Target input length in AISBench string tokens (one word each).")
    p.add_argument(
        "--prompt-tokens-jitter",
        type=int,
        default=0,
        help="Uniform ±jitter on prompt-tokens per request (AISBench string Input range). 0 = identical length.",
    )
    p.add_argument("--max-tokens", type=int, default=128, help="Output tokens; sent with ignore_eos=true.")
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup requests excluded from stats. 0 matches AISBench (first requests are in the table). "
        "Set to concurrency to drop TLS/cold-start from remote-client numbers.",
    )
    p.add_argument(
        "--unique-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, change the first word so prefix cache cannot hit. Default off = AISBench identical 'A A A...'.",
    )
    p.add_argument("--timeout", type=float, default=3600)
    args = p.parse_args()

    if args.concurrency < 1 or args.n_requests < 1:
        print("concurrency and n-requests must be >= 1", file=sys.stderr)
        return 2
    if args.prompt_tokens < 1 or args.max_tokens < 1:
        print("prompt-tokens and max-tokens must be >= 1", file=sys.stderr)
        return 2

    warmup = args.warmup
    if warmup < 0:
        print("warmup must be >= 0", file=sys.stderr)
        return 2
    if args.prompt_tokens_jitter < 0:
        print("prompt-tokens-jitter must be >= 0", file=sys.stderr)
        return 2

    questions: list[str] | None = None
    if args.prompt_source == "gsm8k":
        ds = args.dataset.expanduser() if args.dataset else DEFAULT_GSM8K
        try:
            questions = load_gsm8k_questions(ds)
        except (OSError, ValueError, ImportError, json.JSONDecodeError) as e:
            print(str(e), file=sys.stderr)
            return 2
        def prompt_fn(req_id: int) -> str:
            n_in = args.prompt_tokens
            if args.prompt_tokens_jitter:
                lo = max(1, args.prompt_tokens - args.prompt_tokens_jitter)
                hi = args.prompt_tokens + args.prompt_tokens_jitter
                n_in = random.randint(lo, hi)
            return gsm8k_fixed_prompt(questions, n_in, req_id)
    else:
        def prompt_fn(req_id: int) -> str:
            n_in = args.prompt_tokens
            if args.prompt_tokens_jitter:
                lo = max(1, args.prompt_tokens - args.prompt_tokens_jitter)
                hi = args.prompt_tokens + args.prompt_tokens_jitter
                n_in = random.randint(lo, hi)
            return synthetic_prompt(n_in, req_id, args.unique_prompt)

    # Disable nginx-style SSE buffering on the path (TIONE/gateways).
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    target = parse_target(args.url)

    if args.prompt_source == "gsm8k":
        in_desc = f"gsm8k n={len(questions)} in={args.prompt_tokens}±{args.prompt_tokens_jitter}"
        ds_path = (args.dataset.expanduser() if args.dataset else DEFAULT_GSM8K).resolve()
    else:
        in_desc = f"synthetic in={args.prompt_tokens}±{args.prompt_tokens_jitter}"
        ds_path = None
    print(
        f"POST {target.url}  model={args.model}  "
        f"c={args.concurrency}  n={args.n_requests}  warmup={warmup}  "
        f"{in_desc}  out={args.max_tokens}  unique={args.unique_prompt}",
        flush=True,
    )
    if ds_path is not None:
        print(f"dataset={ds_path}", flush=True)
    if args.unique_prompt:
        print("note: --unique-prompt on → prefix cache miss; TTFT will not match AISBench identical prompts", flush=True)
    if warmup == 0:
        print("note: warmup=0 → first requests (TLS/cold) are in the summary, same as AISBench N=12 tables", flush=True)

    common = dict(
        target=target,
        model=args.model,
        prompt_fn=prompt_fn,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        headers=headers,
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        if warmup:
            run_batch(ex, warmup, start_id=0, label="warmup", **common)
        t_wall0 = time.perf_counter()
        samples = run_batch(
            ex,
            args.n_requests,
            start_id=warmup,
            label="run",
            **common,
        )
        wall = time.perf_counter() - t_wall0

    ok = [s for s in samples if s.ok and s.ttft_sse_s is not None]
    if not ok:
        print("no successful streaming samples", file=sys.stderr)
        return 1

    ttft_sse = [s.ttft_sse_s for s in ok]
    ttft_tok = [s.ttft_token_s for s in ok if s.ttft_token_s is not None]
    tpots = [s.tpot_s for s in ok if s.tpot_s is not None]
    e2es = [s.e2e_s for s in ok]
    out_sum = sum(s.out_tokens for s in ok)
    in_vals = [s.in_tokens for s in ok if s.in_tokens]
    in_mean = statistics.mean(in_vals) if in_vals else 0.0
    out_mean = statistics.mean(s.out_tokens for s in ok)

    def row(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"{name}: n/a")
            return
        print(
            f"{name}: mean={statistics.mean(xs) * 1000:.3f}ms  "
            f"p50={percentile(xs, 50) * 1000:.3f}ms  "
            f"p90={percentile(xs, 90) * 1000:.3f}ms  "
            f"p99={percentile(xs, 99) * 1000:.3f}ms"
        )

    print("---")
    print(
        f"success {len(ok)}/{len(samples)}  wall={wall:.1f}s  "
        f"prompt_tokens(usage)≈{in_mean:.0f}  completion_tokens≈{out_mean:.0f}"
    )
    row("TTFT_sse (AISBench)", ttft_sse)
    row("TTFT_token", ttft_tok)
    row("TPOT (AISBench)", tpots)
    row("E2EL", e2es)
    print(f"output tok/s (system) = {out_sum / wall:.2f}")
    print(f"request TPS           = {len(ok) / wall:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
