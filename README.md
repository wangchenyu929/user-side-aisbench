# user-aisbench

Closed-loop load generator for an OpenAI-compatible chat endpoint (vLLM and similar). It streams each response, records latency the way AISBench does, and can either replay a fixed concurrency matrix or search for the highest concurrency that still meets a TTFT / TPOT limit.

## How a run works

One measured case is a batch of HTTP requests against `POST /v1/chat/completions` with `stream: true`.

1. A thread pool of size `--concurrency` stays full. When a request finishes, the pool starts the next one until `--n-requests` measured requests are done. This is closed-loop load: concurrency is the cap, not an open arrival rate.
2. Optional `--warmup` requests run first on the same pool and are left out of the summary. Use them to drop TLS and cold-start from the numbers. `warmup = 0` keeps those first requests in the table.
3. Each worker reuses one HTTP connection (`keep-alive`, `TCP_NODELAY`). The request sends `X-Accel-Buffering: no` so a reverse proxy is less likely to buffer the SSE stream and inflate time-to-first-token.
4. Generation is pinned with `temperature: 0`, `ignore_eos: true`, and `max_tokens`, so output length is the configured decode length rather than an early stop.

The matrix runner (`src/bench_xlsx_matrix.py`) does not import the client. Every case is a new `python3 src/bench_vllm_curl.py` process. After the case it waits `--cooldown` seconds so the server can drain its queue and KV cache before the next point. That wait is why back-to-back cases stay comparable.

### Prompts

Input length is an AISBench-style string length: one whitespace-separated word stands in for one target token. It is not the tokenizer's token count. The summary still prints `prompt_tokens` from the API `usage` field when the server sends it.

| Source | What is sent |
| --- | --- |
| `synthetic` | `N` copies of the word `A` (`A A A ...`). |
| `synthetic` + unique | Same text, but the first word is `U0`, `U1`, ... so a prefix cache cannot reuse the prefill. |
| `gsm8k` | GSM8K question words from `data/gsm8k.json`, prefixed with `U{id}`, padded with `A` or truncated to `N` words. Every request is unique. |

`--prompt-tokens-jitter` (client only) picks a length uniformly in `[N - jitter, N + jitter]` per request.

### What is measured

Timers use `time.perf_counter()` on the client, from the moment the request is sent.

| Metric | Meaning |
| --- | --- |
| TTFT_sse | Time until the first SSE `data:` line. This is the default SLO series. |
| TTFT_token | Time until the first non-empty content or reasoning delta. |
| E2EL | Time until the last SSE event of that request. |
| TPOT | `(E2EL - TTFT_sse) / (output_tokens - 1)`. Per-token decode time after the first token. Requests with fewer than 2 output tokens are omitted. |
| output tok/s | Sum of completion tokens across successful requests, divided by the measured wall clock. |
| request TPS | Successful requests divided by the same wall clock. |

Each series is reported as mean, p50, p90, and p99 (linear interpolation on the sorted samples).

### Finding a concurrency limit

`mode = auto-probe` searches, per workload `prompt_tokens/max_tokens`, for the largest concurrency that still meets the SLO.

- A point passes only when every measured request succeeds and the chosen statistic (`mean`, `p50`, `p90`, or `p99`) of TTFT and of TPOT is within `--ttft-max-ms` and `--tpot-max-ms`. Set only one limit if you care about a single metric.
- Coarse grid: `1`, then `coarse_step`, `2 * coarse_step`, ... with no preset cap. `n-requests = concurrency * n_multiplier`.
- The first coarse point is never allowed to stop the search. A miss there is treated as cold start, and the next step is still run.
- The first later miss ends the coarse scan. Binary search then fills the gap between the last pass and that failure.
- The workbook records every point and the max passing concurrency.

`mode = matrix` ignores the SLO and runs the fixed table in `CASES` inside `src/bench_xlsx_matrix.py` (TestID 1–12 at 32772/512, 13–18 at 65540/1024). `--test-ids 1,13` or `1-5` selects a subset.

The workbook is rewritten after every case, so a stopped run still has the rows finished so far. The `meta` sheet stores the URL and model from that run.

## Layout

```
config.ini                  # what ./run.sh reads
run.sh                      # builds the matrix command from config.ini
src/bench_vllm_curl.py      # one case: concurrent streaming requests
src/bench_xlsx_matrix.py    # many cases, Excel output, optional SLO search
data/gsm8k.json             # questions used when prompt source is gsm8k
results/                    # created at run time; gitignored
```

## Requirements

Python 3.10+ and `openpyxl` (Excel output only):

```bash
pip install -r requirements.txt
```

## Usage

Edit `config.ini`. Put the real endpoint in `url` and the served model id in `model`. A host, a `/v1` base, or a full `/v1/chat/completions` URL all work. Empty options are omitted. `true` / `false` is case-insensitive.

```bash
chmod +x run.sh
./run.sh
```

Extra arguments are appended to the matrix command:

```bash
./run.sh --test-ids 1,13
./run.sh -c /path/to/other.ini
```

### config.ini

| Section | Keys |
| --- | --- |
| `paths` | `bench_dir` (`.` if this repo is the bench), `python` |
| `run` | `mode`: `auto-probe` or `matrix` |
| `server` | `url`, `model` |
| `prompt` | `source` (`synthetic` or `gsm8k`), `unique`, `dataset` |
| `slo` | `ttft_max_ms`, `tpot_max_ms`, `stat`, `ttft_metric` (`TTFT_sse` or `TTFT_token`) |
| `probe` | `workloads` (`32772/512,65540/1024`), `coarse_step`, `coarse_concurrencies`, `warmup`, `cooldown`, `n_multiplier`, `test_ids` |
| `output` | `out` (xlsx path, relative to `config.ini`) |

`coarse_concurrencies` replaces the start of the coarse grid (`1,5,10,15`). If the last listed point still passes, the search continues in steps of `coarse_step` until a point misses the SLO.

### One case, no Excel

From `src/`:

```bash
python3 bench_vllm_curl.py \
  --url 'https://HOST/v1/chat/completions' \
  --model your-model-name \
  --prompt-source synthetic \
  --concurrency 4 \
  --n-requests 16 \
  --prompt-tokens 1024 \
  --max-tokens 256 \
  --warmup 4
```

### SLO search without config.ini

```bash
python3 src/bench_xlsx_matrix.py \
  --url 'https://HOST/v1' \
  --model your-model-name \
  --auto-probe \
  --ttft-max-ms 2000 \
  --tpot-max-ms 50 \
  --workloads 32772/512,65540/1024 \
  --cooldown 20 \
  --out results/bench.xlsx
```

## Before you push

`config.ini` in this repo uses placeholders. Results workbooks copy the live URL and model into the `meta` sheet, and `results/` is gitignored. Do not commit a filled-in endpoint, a token, or a results file.

This client does not send an `Authorization` header. If the server requires one, add it in `src/bench_vllm_curl.py` locally and keep that change out of the public branch.
