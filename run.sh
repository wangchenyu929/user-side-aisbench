#!/usr/bin/env bash
# Read config.ini and launch the bench. Extra args are appended, e.g. ./run.sh --test-ids 1,13
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$ROOT/config.ini"

if [[ "${1:-}" == "-c" || "${1:-}" == "--config" ]]; then
  CONFIG="$2"
  shift 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

exec python3 - "$ROOT" "$CONFIG" "$@" <<'PY'
from __future__ import annotations

import configparser
import os
import subprocess
import sys
from pathlib import Path


def truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get(cfg: configparser.ConfigParser, section: str, key: str, default: str = "") -> str:
    if not cfg.has_section(section) or not cfg.has_option(section, key):
        return default
    return cfg.get(section, key, fallback=default).strip()


def add(cmd: list[str], flag: str, value: str) -> None:
    if value:
        cmd.extend([flag, value])


root = Path(sys.argv[1]).resolve()
cfg_path = Path(sys.argv[2]).expanduser()
if not cfg_path.is_absolute():
    cfg_path = (root / cfg_path).resolve()
extra = sys.argv[3:]

cfg = configparser.ConfigParser(interpolation=None)
read = cfg.read(cfg_path, encoding="utf-8")
if not read:
    sys.exit(f"failed to read {cfg_path}")

bench_dir = Path(get(cfg, "paths", "bench_dir", "user-aisbench"))
if not bench_dir.is_absolute():
    bench_dir = (root / bench_dir).resolve()
src = bench_dir / "src"
matrix = src / "bench_xlsx_matrix.py"
curl = src / "bench_vllm_curl.py"
if not matrix.is_file() or not curl.is_file():
    sys.exit(f"bench scripts not found under {src}")

py = get(cfg, "paths", "python", "python3") or "python3"
mode = get(cfg, "run", "mode", "auto-probe").lower()
url = get(cfg, "server", "url")
model = get(cfg, "server", "model")
source = get(cfg, "prompt", "source", "synthetic")
dataset = get(cfg, "prompt", "dataset")
out = get(cfg, "output", "out")

if out:
    out_path = Path(out).expanduser()
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = str(out_path)

if dataset:
    ds_path = Path(dataset).expanduser()
    if not ds_path.is_absolute():
        ds_path = (root / ds_path).resolve()
    dataset = str(ds_path)

os.chdir(src)

cmd = [
    py,
    str(matrix),
    "--url",
    url,
    "--model",
    model,
    "--prompt-source",
    source,
    "--warmup",
    get(cfg, "probe", "warmup", "1") or "1",
    "--cooldown",
    get(cfg, "probe", "cooldown", "20") or "20",
    "--slo-stat",
    get(cfg, "slo", "stat", "mean") or "mean",
    "--ttft-metric",
    get(cfg, "slo", "ttft_metric", "TTFT_sse") or "TTFT_sse",
    "--coarse-step",
    get(cfg, "probe", "coarse_step", "5") or "5",
    "--n-multiplier",
    get(cfg, "probe", "n_multiplier", "4") or "4",
    "--python",
    py,
]
add(cmd, "--dataset", dataset)
add(cmd, "--out", out)
add(cmd, "--test-ids", get(cfg, "probe", "test_ids"))
add(cmd, "--coarse-concurrencies", get(cfg, "probe", "coarse_concurrencies"))
if truthy(get(cfg, "prompt", "unique")):
    cmd.append("--unique-prompt")
if mode in {"auto-probe", "auto_probe", "probe"}:
    cmd.append("--auto-probe")
    add(cmd, "--ttft-max-ms", get(cfg, "slo", "ttft_max_ms"))
    add(cmd, "--tpot-max-ms", get(cfg, "slo", "tpot_max_ms"))
    add(cmd, "--workloads", get(cfg, "probe", "workloads"))
elif mode not in {"matrix", "fixed", "fixed-matrix"}:
    sys.exit(f"unknown run.mode={mode!r} (use auto-probe / matrix)")

cmd.extend(extra)
print("config=", cfg_path)
print("cmd=", " ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
PY
