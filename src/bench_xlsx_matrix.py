#!/usr/bin/env python3
"""Run each matrix case as a fresh `python3 bench_vllm_curl.py` process.

Does not import or reuse the bench module. Each case is the same command you
run by hand; only concurrency / n-requests / prompt-tokens / max-tokens change.

  python3 bench_vllm_curl.py \
    --url http://HOST/v1 \
    --model your-model-name \
    --concurrency 1 --n-requests 4 \
    --prompt-tokens 32772 --max-tokens 512

Manual cases are TestID 1-18 in CASES: 1-12 use 32772/512, 13-18 use 65540/1024.

Auto-probe finds the largest concurrency that still meets TTFT and TPOT limits:
  coarse grid first (1, 5, 10, 15, ...) until a point misses the SLO, then binary
  search between the last passing and first failing point. The first coarse
  concurrency (usually 1) is always followed by the next (usually 5), even if
  it misses the SLO, so model-level cold start cannot abort the probe.

Results go to Excel (--out). The file is saved after every case.

Why a previous matrix run looked worse than a single command:
  The wrapper used bench_vllm_curl_adjusted.py, which fires extra 32k-token
  calibrate probes before the measured batch, then immediately starts the next
  case. Residual KV / queueing inflates TTFT and TPOT. This wrapper calls
  bench_vllm_curl.py in a new process and waits --cooldown seconds between cases.

Example:

  python3 bench_xlsx_matrix.py \\
    --url http://HOST/v1 \\
    --model your-model-name

  python3 bench_xlsx_matrix.py ... --test-ids 1,13
  python3 bench_xlsx_matrix.py ... --cooldown 60

  python3 bench_xlsx_matrix.py ... --auto-probe \\
    --ttft-max-ms 2000 --tpot-max-ms 50 \\
    --workloads 32772/512,65540/1024

  python3 bench_xlsx_matrix.py ... --auto-probe --prompt-source gsm8k \\
    --ttft-max-ms 2000 --tpot-max-ms 50 --workloads 32772/512
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = HERE / "bench_vllm_curl.py"
DEFAULT_GSM8K = ROOT / "data" / "gsm8k.json"
DEFAULT_OUT = ROOT / "results" / "bench_xlsx_matrix_results.xlsx"

# TestID, n-requests, concurrency, prompt-tokens, max-tokens
CASES: list[tuple[int, int, int, int, int]] = [
    (1, 4, 1, 32772, 512),
    (2, 8, 2, 32772, 512),
    (3, 12, 3, 32772, 512),
    (4, 16, 4, 32772, 512),
    (5, 20, 5, 32772, 512),
    (6, 24, 6, 32772, 512),
    (7, 28, 7, 32772, 512),
    (8, 28, 8, 32772, 512),
    (9, 40, 10, 32772, 512),
    (10, 60, 15, 32772, 512),
    (11, 68, 17, 32772, 512),
    (12, 80, 20, 32772, 512),
    (13, 4, 1, 65540, 1024),
    (14, 8, 2, 65540, 1024),
    (15, 12, 3, 65540, 1024),
    (16, 16, 4, 65540, 1024),
    (17, 20, 5, 65540, 1024),
    (18, 40, 10, 65540, 1024),
]

DEFAULT_WORKLOADS: list[tuple[int, int]] = [
    (32772, 512),
    (65540, 1024),
]

COLUMNS = [
    "TestID",
    "TotalRequests",
    "MaxConcurrency",
    "PromptTokens",
    "MaxTokens",
    "ExitCode",
    "SuccessRequests",
    "FailedRequests",
    "Wall_s",
    "PromptTokens_usage",
    "CompletionTokens",
    "TTFT_sse_mean_ms",
    "TTFT_sse_p50_ms",
    "TTFT_sse_p90_ms",
    "TTFT_sse_p99_ms",
    "TTFT_token_mean_ms",
    "TTFT_token_p50_ms",
    "TTFT_token_p90_ms",
    "TTFT_token_p99_ms",
    "TPOT_mean_ms",
    "TPOT_p50_ms",
    "TPOT_p90_ms",
    "TPOT_p99_ms",
    "E2EL_mean_ms",
    "E2EL_p50_ms",
    "E2EL_p90_ms",
    "E2EL_p99_ms",
    "OutputTok_s",
    "RequestTPS",
]

STAT_LINE = re.compile(
    r"^(?P<name>.+?): "
    r"mean=(?P<mean>[0-9.]+)ms\s+"
    r"p50=(?P<p50>[0-9.]+)ms\s+"
    r"p90=(?P<p90>[0-9.]+)ms\s+"
    r"p99=(?P<p99>[0-9.]+)ms\s*$"
)
STAT_NAME_KEYS = {
    "TTFT_sse (AISBench)": "TTFT_sse",
    "TTFT_sse": "TTFT_sse",
    "TTFT_token": "TTFT_token",
    "TPOT (AISBench)": "TPOT",
    "TPOT": "TPOT",
    "E2EL": "E2EL",
}
SUCCESS_LINE = re.compile(
    r"success\s+(?P<ok>\d+)/(?P<n>\d+)\s+"
    r"wall=(?P<wall>[0-9.]+)s\s+"
    r"prompt_tokens\(usage\)≈(?P<pin>[0-9.]+)\s+"
    r"completion_tokens≈(?P<pout>[0-9.]+)"
)
OUT_TPS = re.compile(r"output tok/s \(system\)\s*=\s*(?P<v>[0-9.]+)")
REQ_TPS = re.compile(r"request TPS\s*=\s*(?P<v>[0-9.]+)")
FAIL_LINE = re.compile(r"FAIL\s+(.+)$")
SLO_STATS = ("mean", "p50", "p90", "p99")


def parse_test_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.add(int(part))
    return out


def parse_int_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise argparse.ArgumentTypeError(f"concurrency must be >= 1, got {n}")
        if n not in out:
            out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("empty concurrency list")
    return out


def parse_workloads(raw: str | None, prompt_source: str = "synthetic") -> list[tuple[int, int]]:
    del prompt_source
    if not raw:
        return list(DEFAULT_WORKLOADS)
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "/" not in part:
            raise argparse.ArgumentTypeError(
                f"workload must be prompt/max-tokens, got {part!r}"
            )
        a, b = part.split("/", 1)
        prompt, max_tokens = int(a.strip()), int(b.strip())
        if prompt < 1 or max_tokens < 1:
            raise argparse.ArgumentTypeError(f"invalid workload {part!r}")
        out.append((prompt, max_tokens))
    if not out:
        raise argparse.ArgumentTypeError("empty --workloads")
    return out


def _f(v: str | None) -> float | None:
    if v is None:
        return None
    return float(v)


def parse_bench_stdout(text: str, exit_code: int) -> dict:
    row: dict = {"ExitCode": exit_code}
    errs: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        sm = SUCCESS_LINE.search(s)
        if sm:
            ok = int(sm.group("ok"))
            n = int(sm.group("n"))
            row["SuccessRequests"] = ok
            row["FailedRequests"] = n - ok
            row["Wall_s"] = _f(sm.group("wall"))
            row["PromptTokens_usage"] = _f(sm.group("pin"))
            row["CompletionTokens"] = _f(sm.group("pout"))
            continue
        st = STAT_LINE.match(s)
        if st:
            key = STAT_NAME_KEYS.get(st.group("name"))
            if key:
                row[f"{key}_mean_ms"] = _f(st.group("mean"))
                row[f"{key}_p50_ms"] = _f(st.group("p50"))
                row[f"{key}_p90_ms"] = _f(st.group("p90"))
                row[f"{key}_p99_ms"] = _f(st.group("p99"))
            continue
        om = OUT_TPS.search(s)
        if om:
            row["OutputTok_s"] = _f(om.group("v"))
        rm = REQ_TPS.search(s)
        if rm:
            row["RequestTPS"] = _f(rm.group("v"))
        fm = FAIL_LINE.search(s)
        if fm:
            errs.append(fm.group(1).strip()[:240])
    error = "; ".join(dict.fromkeys(errs))[:1000]
    if exit_code != 0 and not error:
        tail = [ln.strip() for ln in text.splitlines() if ln.strip()][-8:]
        error = " | ".join(tail)[:1000]
    row["Error"] = error
    return row


def _style_header(ws, headers: list[str]) -> None:
    fill = PatternFill("solid", fgColor="1F4E79")
    font = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
    align = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[1].height = 32
    for col, name in enumerate(headers, 1):
        cell = ws.cell(1, col, name)
        cell.fill = fill
        cell.font = font
        cell.alignment = align
        width = 48 if name == "Command" else min(max(len(name) + 2, 12), 24)
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"


def save_workbook(path: Path, rows: list[dict], meta: dict) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    _style_header(ws, COLUMNS)
    for r_i, row in enumerate(rows, 2):
        for c_i, name in enumerate(COLUMNS, 1):
            ws.cell(r_i, c_i, row.get(name))
    if rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{1 + len(rows)}"

    meta_ws = wb.create_sheet("meta")
    meta_ws.cell(1, 1, "key")
    meta_ws.cell(1, 2, "value")
    for i, (k, v) in enumerate(meta.items(), 2):
        meta_ws.cell(i, 1, k)
        meta_ws.cell(i, 2, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def build_cmd(
    python: str,
    url: str,
    model: str,
    conc: int,
    n_requests: int,
    prompt_tokens: int,
    max_tokens: int,
    extra: list[str],
) -> list[str]:
    cmd = [
        python,
        str(BENCH),
        "--url",
        url,
        "--model",
        model,
        "--concurrency",
        str(conc),
        "--n-requests",
        str(n_requests),
        "--max-tokens",
        str(max_tokens),
    ]
    if prompt_tokens and prompt_tokens > 0:
        cmd.extend(["--prompt-tokens", str(prompt_tokens)])
    cmd.extend(extra)
    return cmd


def run_isolated(cmd: list[str]) -> tuple[int, str]:
    """One OS process per case, new session, cwd = script dir (same as a shell cd)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(HERE),
        env=os.environ.copy(),
        start_new_session=True,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    rc = proc.wait()
    return rc, "".join(chunks)


def workload_name(prompt_tokens: int, max_tokens: int) -> str:
    return f"{prompt_tokens}/{max_tokens}"


def n_requests_for(conc: int, multiplier: int) -> int:
    return max(conc * multiplier, 1)


def metric_key(prefix: str, stat: str) -> str:
    return f"{prefix}_{stat}_ms"


def slo_verdict(
    row: dict,
    *,
    ttft_max_ms: float | None,
    tpot_max_ms: float | None,
    stat: str,
    ttft_prefix: str,
) -> tuple[bool, str, float | None, float | None]:
    """Return (meets, reason, compared_ttft_ms, compared_tpot_ms)."""
    reasons: list[str] = []
    total = (row.get("SuccessRequests") or 0) + (row.get("FailedRequests") or 0)
    ok_n = row.get("SuccessRequests") or 0
    if row.get("ExitCode") not in (0, None):
        reasons.append(f"exit={row.get('ExitCode')}")
    if total <= 0 or ok_n <= 0:
        reasons.append("no_success")
    elif ok_n < total:
        reasons.append(f"success_rate={ok_n}/{total}")

    ttft = row.get(metric_key(ttft_prefix, stat))
    tpot = row.get(metric_key("TPOT", stat))
    if ttft_max_ms is not None:
        if ttft is None:
            reasons.append("ttft_missing")
        elif ttft > ttft_max_ms:
            reasons.append(f"TTFT_{stat}={ttft:.3f}>{ttft_max_ms:g}")
    if tpot_max_ms is not None:
        if tpot is None:
            reasons.append("tpot_missing")
        elif tpot > tpot_max_ms:
            reasons.append(f"TPOT_{stat}={tpot:.3f}>{tpot_max_ms:g}")

    if reasons:
        return False, "; ".join(reasons), ttft, tpot
    return True, "", ttft, tpot


def iter_coarse(step: int, preset: list[int] | None):
    """Yield 1, step, 2*step, ... with no upper bound. Stop is the SLO, not a cap."""
    if step < 1:
        raise ValueError("coarse step must be >= 1")
    seen: set[int] = set()
    last = 0
    if preset:
        for c in preset:
            if c < 1 or c in seen:
                continue
            seen.add(c)
            last = c
            yield c
    if last == 0:
        seen.add(1)
        yield 1
        last = 1
    nxt = step if last == 1 and step > 1 else last + step
    while True:
        if nxt not in seen:
            seen.add(nxt)
            yield nxt
        nxt += step


def pick_refine_mid(lo: int, hi: int, tested: set[int]) -> int | None:
    candidates = [c for c in range(lo + 1, hi) if c not in tested]
    if not candidates:
        return None
    target = (lo + hi) // 2
    return min(candidates, key=lambda c: (abs(c - target), c))


class RunContext:
    def __init__(
        self,
        *,
        python: str,
        url: str,
        model: str,
        extra: list[str],
        cooldown: float,
        out: Path,
        started: str,
        meta_extra: dict,
    ) -> None:
        self.python = python
        self.url = url
        self.model = model
        self.extra = extra
        self.cooldown = cooldown
        self.out = out
        self.started = started
        self.meta_extra = meta_extra
        self.rows: list[dict] = []
        self.summaries: list[dict] = []
        self.failed: list[int] = []
        self.next_id = 1

    def save(self, updated_at: str, done_label: str) -> None:
        meta = {
            "started_at": self.started,
            "updated_at": updated_at,
            "url": self.url,
            "model": self.model,
            "cooldown_s": self.cooldown,
            "done_cases": done_label,
            "bench": str(BENCH),
            "note": "Each case is a new python3 bench_vllm_curl.py process; no adjusted calib/RTT probes.",
        }
        meta.update(self.meta_extra)
        save_workbook(self.out, self.rows, meta)

    def run_case(
        self,
        *,
        conc: int,
        n_requests: int,
        prompt_tokens: int,
        max_tokens: int,
        label: str,
        workload: str | None = None,
        phase: str | None = None,
        slo: dict | None = None,
        cooldown_after: bool = True,
    ) -> dict:
        test_id = self.next_id
        self.next_id += 1
        cmd = build_cmd(
            self.python,
            self.url,
            self.model,
            conc,
            n_requests,
            prompt_tokens,
            max_tokens,
            self.extra,
        )
        print(
            f"\n=== [{label}] isolated python  TestID={test_id}  "
            f"n={n_requests} c={conc} {prompt_tokens}/{max_tokens} ===",
            flush=True,
        )
        print("    " + " ".join(cmd), flush=True)
        rc, text = run_isolated(cmd)
        finished = dt.datetime.now().isoformat(timespec="seconds")
        parsed = parse_bench_stdout(text, rc)
        parsed.update(
            {
                "TestID": test_id,
                "TotalRequests": n_requests,
                "MaxConcurrency": conc,
                "PromptTokens": prompt_tokens,
                "MaxTokens": max_tokens,
                "FinishedAt": finished,
                "Command": " ".join(cmd),
                "Workload": workload or workload_name(prompt_tokens, max_tokens),
                "ProbePhase": phase,
            }
        )
        if slo:
            meets, reason, ttft, tpot = slo_verdict(parsed, **slo)
            parsed["MeetsSLO"] = "Y" if meets else "N"
            parsed["FailReason"] = reason
            parsed["ComparedTTFT_ms"] = ttft
            parsed["ComparedTPOT_ms"] = tpot
            print(
                f"  SLO {'PASS' if meets else 'FAIL'}  conc={conc}  "
                f"TTFT={ttft if ttft is not None else '-'}ms  "
                f"TPOT={tpot if tpot is not None else '-'}ms"
                + (f"  ({reason})" if reason else ""),
                flush=True,
            )
        self.rows.append(parsed)
        self.save(finished, label)
        print(f"  saved TestID={test_id} → {self.out} !results", flush=True)
        if rc != 0:
            self.failed.append(test_id)
            print(f"TestID={test_id} exited {rc}", file=sys.stderr)
        if cooldown_after and self.cooldown > 0:
            print(f"  cooldown {self.cooldown:.0f}s before next isolated python ...", flush=True)
            time.sleep(self.cooldown)
        return parsed


def probe_one_workload(
    ctx: RunContext,
    *,
    prompt_tokens: int,
    max_tokens: int,
    coarse_step: int,
    coarse_preset: list[int] | None,
    n_multiplier: int,
    slo: dict,
    workload_i: int,
    workload_n: int,
) -> dict:
    name = workload_name(prompt_tokens, max_tokens)
    last_pass: int | None = None
    first_fail: int | None = None
    tested: set[int] = set()
    pass_rows: dict[int, dict] = {}
    coarse_tested: list[int] = []

    print(
        f"\n######## workload {workload_i}/{workload_n}  {name}  "
        f"coarse=1,{coarse_step},{2 * coarse_step},... "
        f"(first coarse never stops the search) ########",
        flush=True,
    )

    for conc in iter_coarse(coarse_step, coarse_preset):
        if first_fail is not None:
            break
        if conc in tested:
            continue
        row = ctx.run_case(
            conc=conc,
            n_requests=n_requests_for(conc, n_multiplier),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            label=f"probe {name} coarse c={conc}",
            workload=name,
            phase="coarse",
            slo=slo,
        )
        tested.add(conc)
        coarse_tested.append(conc)
        if row.get("MeetsSLO") == "Y":
            last_pass = conc
            pass_rows[conc] = row
        elif len(coarse_tested) == 1:
            print(
                f"  first coarse c={conc} missed SLO (likely model cold start); "
                f"continue to next coarse, do not stop",
                flush=True,
            )
        else:
            first_fail = conc
            break

    while first_fail is not None:
        lo = last_pass or 0
        hi = first_fail
        mid = pick_refine_mid(lo, hi, tested)
        if mid is None:
            break
        row = ctx.run_case(
            conc=mid,
            n_requests=n_requests_for(mid, n_multiplier),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            label=f"probe {name} refine c={mid} ({lo}<c<{hi})",
            workload=name,
            phase="refine",
            slo=slo,
        )
        tested.add(mid)
        if row.get("MeetsSLO") == "Y":
            last_pass = mid
            pass_rows[mid] = row
        else:
            first_fail = mid

    best = pass_rows.get(last_pass) if last_pass is not None else None
    summary = {
        "Workload": name,
        "PromptTokens": prompt_tokens,
        "MaxTokens": max_tokens,
        "MaxPassingConcurrency": last_pass if last_pass is not None else 0,
        "FirstFailingConcurrency": first_fail,
        "SLO_TTFT_max_ms": slo["ttft_max_ms"],
        "SLO_TPOT_max_ms": slo["tpot_max_ms"],
        "SLOStat": slo["stat"],
        "TTFT_at_max_ms": None if best is None else best.get("ComparedTTFT_ms"),
        "TPOT_at_max_ms": None if best is None else best.get("ComparedTPOT_ms"),
        "CoarsePoints": ",".join(str(c) for c in coarse_tested),
        "TestedConcurrencies": ",".join(str(c) for c in sorted(tested)),
    }
    ctx.summaries.append(summary)
    print(
        f"\n  workload {name}: max passing conc={summary['MaxPassingConcurrency']}  "
        f"first fail={first_fail}  tested=[{summary['TestedConcurrencies']}]",
        flush=True,
    )
    ctx.save(dt.datetime.now().isoformat(timespec="seconds"), f"probe {name} done")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--test-ids", default=None, help="Manual-mode subset, e.g. 1,13 or 1-5,13-17")
    p.add_argument(
        "--cooldown",
        type=float,
        default=20.0,
        help="Idle seconds after each case so the next python process hits a drained engine. 0 disables.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Requests fired before stats/SLO. Default 1 drops the cold first request. 0 keeps it in the table.",
    )
    p.add_argument("--unique-prompt", action="store_true")
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
        help="GSM8K JSON. Default: user-aisbench/data/gsm8k.json.",
    )
    p.add_argument("--python", default="python3", help="Interpreter used for each isolated command.")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Excel path for parsed case summaries.",
    )
    p.add_argument(
        "--auto-probe",
        action="store_true",
        help="Search max concurrency under TTFT/TPOT limits instead of the fixed TestID matrix.",
    )
    p.add_argument("--ttft-max-ms", type=float, default=None, help="TTFT SLO in milliseconds.")
    p.add_argument("--tpot-max-ms", type=float, default=None, help="TPOT SLO in milliseconds.")
    p.add_argument(
        "--slo-stat",
        choices=SLO_STATS,
        default="mean",
        help="Which TTFT/TPOT statistic must stay under the limits. Default: mean.",
    )
    p.add_argument(
        "--ttft-metric",
        choices=("TTFT_sse", "TTFT_token"),
        default="TTFT_sse",
        help="Which TTFT series to compare against --ttft-max-ms.",
    )
    p.add_argument(
        "--workloads",
        default=None,
        help="Probe workloads as prompt/max-tokens, comma-separated. Default: 32772/512,65540/1024.",
    )
    p.add_argument(
        "--coarse-step",
        type=int,
        default=5,
        help="Coarse concurrency stride after 1. Default 5 → 1,5,10,15,...",
    )
    p.add_argument(
        "--coarse-concurrencies",
        default=None,
        help="Override coarse starting list, e.g. 1,5,10,15. If the last still passes, continue with --coarse-step until SLO fail.",
    )
    p.add_argument(
        "--n-multiplier",
        type=int,
        default=4,
        help="n-requests = concurrency * this. Default 4, same as the fixed matrix.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not BENCH.is_file():
        print(f"bench script not found: {BENCH}", file=sys.stderr)
        return 2
    if args.n_multiplier < 1:
        print("--n-multiplier must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2
    if args.auto_probe:
        if args.ttft_max_ms is None and args.tpot_max_ms is None:
            print("--auto-probe needs --ttft-max-ms and/or --tpot-max-ms", file=sys.stderr)
            return 2
        if args.test_ids:
            print("--test-ids is for the fixed matrix; do not combine with --auto-probe", file=sys.stderr)
            return 2
        if args.coarse_step < 1:
            print("--coarse-step must be >= 1", file=sys.stderr)
            return 2
        try:
            workloads = parse_workloads(args.workloads, args.prompt_source)
            coarse_preset = parse_int_list(args.coarse_concurrencies)
        except (argparse.ArgumentTypeError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
    else:
        workloads = []
        coarse_preset = None

    if args.prompt_source == "gsm8k":
        ds = (args.dataset or DEFAULT_GSM8K).expanduser().resolve()
        if not ds.is_file():
            print(f"GSM8K data not found: {ds}", file=sys.stderr)
            return 2

    extra: list[str] = ["--prompt-source", args.prompt_source, "--warmup", str(args.warmup)]
    if args.prompt_source == "gsm8k":
        ds = (args.dataset or DEFAULT_GSM8K).expanduser().resolve()
        extra.extend(["--dataset", str(ds)])
    if args.unique_prompt:
        extra.append("--unique-prompt")

    out = args.out.expanduser().resolve()
    print(f"bench={BENCH}")
    print(f"url={args.url}  model={args.model}  cooldown={args.cooldown}s")
    print(f"prompt-source={args.prompt_source}  warmup={args.warmup}  out={out}")

    if args.auto_probe:
        print(
            f"mode=auto-probe  ttft_max={args.ttft_max_ms}ms  tpot_max={args.tpot_max_ms}ms  "
            f"stat={args.slo_stat}  ttft_metric={args.ttft_metric}"
        )
        print(f"workloads={[workload_name(a, b) for a, b in workloads]}")
        start = ",".join(str(c) for c in coarse_preset) if coarse_preset else f"1,{args.coarse_step},{2 * args.coarse_step}"
        print(
            f"coarse={start},... step={args.coarse_step} until SLO fail  "
            f"n-requests={args.n_multiplier}*concurrency"
        )

        slo = dict(
            ttft_max_ms=args.ttft_max_ms,
            tpot_max_ms=args.tpot_max_ms,
            stat=args.slo_stat,
            ttft_prefix=args.ttft_metric,
        )
        ctx = RunContext(
            python=args.python,
            url=args.url,
            model=args.model,
            extra=extra,
            cooldown=args.cooldown,
            out=out,
            started=dt.datetime.now().isoformat(timespec="seconds"),
            meta_extra={
                "mode": "auto-probe",
                "ttft_max_ms": args.ttft_max_ms,
                "tpot_max_ms": args.tpot_max_ms,
                "slo_stat": args.slo_stat,
                "ttft_metric": args.ttft_metric,
                "coarse_step": args.coarse_step,
                "coarse_preset": ",".join(str(c) for c in coarse_preset) if coarse_preset else "",
                "n_multiplier": args.n_multiplier,
                "prompt_source": args.prompt_source,
                "warmup": args.warmup,
            },
        )
        for i, (prompt_tokens, max_tokens) in enumerate(workloads, 1):
            probe_one_workload(
                ctx,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                coarse_step=args.coarse_step,
                coarse_preset=coarse_preset,
                n_multiplier=args.n_multiplier,
                slo=slo,
                workload_i=i,
                workload_n=len(workloads),
            )
        if ctx.failed:
            print(f"failed TestIDs: {ctx.failed}", file=sys.stderr)
            print(f"wrote {len(ctx.rows)} rows → {out}")
            return 1
        print(f"\nall probe workloads finished → {out}")
        for s in ctx.summaries:
            print(
                f"  {s['Workload']}: max_passing={s['MaxPassingConcurrency']}  "
                f"first_fail={s['FirstFailingConcurrency']}  tested={s['TestedConcurrencies']}"
            )
        return 0

    want = parse_test_ids(args.test_ids)
    cases = [c for c in CASES if want is None or c[0] in want]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 2
    print(f"mode=fixed-matrix  cases={len(cases)}")
    cmds: list[list[str]] = []
    for test_id, n_requests, conc, prompt_tokens, max_tokens in cases:
        cmd = build_cmd(
            args.python,
            args.url,
            args.model,
            conc,
            n_requests,
            prompt_tokens,
            max_tokens,
            extra,
        )
        cmds.append(cmd)
        print(
            f"  TestID={test_id}  n={n_requests}  conc={conc}  "
            f"in={prompt_tokens}  out={max_tokens}"
        )
        print("    " + " ".join(cmd))

    ctx = RunContext(
        python=args.python,
        url=args.url,
        model=args.model,
        extra=extra,
        cooldown=args.cooldown,
        out=out,
        started=dt.datetime.now().isoformat(timespec="seconds"),
        meta_extra={"mode": "fixed-matrix", "prompt_source": args.prompt_source},
    )
    # Keep original TestID values from CASES.
    ctx.next_id = cases[0][0]
    for i, ((test_id, n_requests, conc, prompt_tokens, max_tokens), cmd) in enumerate(
        zip(cases, cmds), 1
    ):
        ctx.next_id = test_id
        ctx.run_case(
            conc=conc,
            n_requests=n_requests,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            label=f"{i}/{len(cases)}",
            cooldown_after=i < len(cases),
        )

    if ctx.failed:
        print(f"failed TestIDs: {ctx.failed}", file=sys.stderr)
        print(f"wrote {len(ctx.rows)} rows → {out}")
        return 1
    print(f"\nall {len(cases)} cases finished → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
