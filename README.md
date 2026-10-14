# kev-rust

**Kev-0.8B inference in pure Rust on [Candle](https://github.com/huggingface/candle). One binary, CPU or CUDA, no Python at runtime.**

[Jared Palmer's Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) answers structured
questions about a state (`choice`, `score`, `noul`). This crate reimplements its Qwen3.5
backbone and pointer head natively, and checks every answer against the official Python model.

[**Interactive benchmark explorer and request builder**](docs/index.html)

```bash
echo '{"state":"I was charged twice. Please refund the duplicate payment.","questions":{"topic":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"charges and refunds","shipping":"delivery and tracking","technical":"bugs and outages"}}}}' \
  | target/release/kev-rust
```

## Why trust it

| Check | Result |
| --- | --- |
| Token IDs and option order vs. official Python fp32 | identical on 22 questions |
| Decision (argmax) flips | **0** on CPU and CUDA |
| Max probability error | `1.2e-6` CPU, `4.9e-5` CUDA (Python rounds its output) |
| Precision | fp32 throughout, no quantization, TF32 off |
| Unit tests | scan and convolution kernels compared with tensor references, CPU and CUDA |

Parity data, logs and the exact pinned checkpoint, Candle and Kev revisions are in
[`results/`](results/) and [`BENCHMARK.md`](BENCHMARK.md). Reproduce with `scripts/validate.py`.

## Speed

Warmed median, fp32. Hardware: two-vCPU Xeon (CPU) and Tesla T4 (GPU) on Google Colab.

| Setup | Median |
| --- | ---: |
| CPU, Python fp32 (reference) | 3.38 s |
| CPU, Rust shared-prefix batched | 3.47 s |
| CPU, Rust shared-prefix batched (GPU-job VM) | 2.87 s |
| T4 CUDA, Rust shared-prefix batched | **89.6 ms** |

- **Shared-prefix batching is the real win**: the state is encoded once and reused by all
  questions. 1.63x faster than recomputing full rows on CPU; 3.3x on a 16-question GPU request.
- **Rust is not faster than PyTorch at warmed GPU compute.** The two are within roughly
  +/-20% depending on scenario; PyTorch wins on a ~600-token context. The case for Rust here is
  deployment, not raw speed: a single binary, and a reported cold start of 119 ms vs 1.5 s.
- Small sample (16 warmed samples per mode, short requests, one run each). Treat as a
  probe, not a production claim. Other CPUs, longer inputs and thread counts are untested.

## Build

```bash
cargo build --release                  # CPU
cargo build --release --features cuda  # needs nvcc
```

Export the checkpoint once (needs Python), then run:

```bash
uv venv --python 3.13 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/prepare.py
target/release/kev-rust --input fixtures/requests.jsonl
```

Everything below is the detailed reference.

---

## Run in Colab on CPU

The Colab job runs all builds, inference tests, and benchmarks remotely. It requests
a standard **CPU** runtime; no GPU or TPU flags are used. Python installs the CPU
PyTorch wheel, and every inference path explicitly uses CPU.

From the Candle repository, prepare a source-only archive and launch it with the
official Google Colab CLI:

```bash
tar -czf /tmp/kev-rust-colab.tar.gz --exclude=target --exclude=.venv \
  --exclude=weights --exclude=research --exclude=results --exclude=dist --exclude=__pycache__ \
  -C experiments/kev-rust .
colab new -s kev-cpu
colab upload -s kev-cpu /tmp/kev-rust-colab.tar.gz /content/kev-rust-colab.tar.gz
colab upload -s kev-cpu experiments/kev-rust/colab/job.py /content/kev_cpu_job.py
colab exec -s kev-cpu --env KEV_COLAB_EXPERIMENT=1 -f experiments/kev-rust/colab/start.py
colab exec -s kev-cpu -f experiments/kev-rust/colab/poll.py
```

Poll the same job until its status is `complete`. Then retrieve the results and
release the runtime:

```bash
colab download -s kev-cpu /content/kev-cpu-results.tar.gz /tmp/kev-cpu-results.tar.gz
colab download -s kev-cpu /content/kev-cpu-job.log /tmp/kev-cpu-job.log
colab stop -s kev-cpu
```

The job fetches pinned Candle and Kev commits, exports the pinned checkpoint,
checks three native modes against the official Python implementation, and runs
five timing repetitions per benchmark request with two threads on both sides.
Logs and status remain in `/content` while the runtime is active. Preserve the
source archive with the report to identify the exact uncommitted experiment code.

## Prepare and run

From this repository:

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/prepare.py
cargo build --release
RAYON_NUM_THREADS=2 target/release/kev-rust --input fixtures/requests.jsonl
```

The exporter downloads pinned base and adapter revisions, checks that every LoRA
tensor was applied, and writes a text-only fp32 checkpoint into `weights/`. Downloads
use the Hugging Face cache. Weights, virtual environments, and build outputs are ignored
by Git. Preserve `metadata.json` alongside the exported weights.

The default runner loads the exported backbone and pointer head, computes the shared
state once, and batches independent question rows against copies of the prefix caches.
CPU DeltaNet and depthwise convolution use fused Rust loops.
CUDA uses fused GPU kernels for those operations and retains model weights and
recurrent caches on the GPU. Both paths retain fp32 precision throughout; TF32
matrix multiplication is disabled.

## Run on an NVIDIA GPU

Use the same exported `weights/` directory as CPU inference. Build with a CUDA
toolkit containing `nvcc`, then select the GPU explicitly:

```bash
cargo build --release --features cuda
target/release/kev-rust --device cuda --weights weights --input fixtures/requests.jsonl
```

The CUDA-enabled binary also supports `--device cpu` (the default).
`--device-index 1` selects a different GPU. A CPU-only build reports an error when
CUDA is requested. GPU inference uses no Python runtime and does not copy the
DeltaNet recurrence to CPU. Timings synchronize CUDA before and after inference.

The distributed CUDA binary targets Linux x86-64 and requires NVIDIA drivers and
CUDA runtime libraries. Its `build.json` records the toolkit and linked libraries;
build on your target system when its toolkit differs. Custom kernels emit PTX for
compute capability 7.5 or newer. No quantization or fp16 conversion is applied.
The downloaded build uses the repository's `target-cpu=native` setting on Colab's
Intel Xeon Skylake CPU, including AVX-512. Rebuild for a different CPU instead of
assuming the executable is portable across all x86-64 hosts.

For Colab, upload a source bundle as `/content/kev-gpu-source.tar.gz` and launch
`colab/gpu_job.py` in a GPU runtime. The bundle contains `src`, `cuda`, `build.rs`,
`Cargo.toml`, `Cargo.lock`, `scripts`, `fixtures`, and the saved official Python
FP32 parity output as `reference-cpu.jsonl`. The job uses its own directory at
`/content/candle/experiments/kev-rust-gpu`, runs CPU/CUDA kernel tests, exports the
weights, checks three CUDA execution modes, benchmarks CPU and GPU, and packages
the executable before completion. Download `/content/kev-gpu-results.tar.gz` and
`/content/kev-rust-linux-x86_64-cuda.tar.gz` before releasing a runtime you own.

Read one request per line from stdin, or use `--input`. Options:

| Option | Behavior |
| --- | --- |
| `--weights DIR` | Exported checkpoint directory, default `weights` |
| `--device cpu\|cuda` | Inference backend, default CPU; CUDA requires the CUDA build |
| `--device-index N` | GPU ordinal, default 0 |
| `--rows` | Recompute the full state per question (comparison baseline) |
| `--tensor` | Use the original allocating tensor scan and grouped convolution |
| `--unbatched` | Run question rows separately for comparison |
| `--batch` | Explicitly request batching; enabled by default with fused kernels |
| `--repeats N` | Run each request N times with weights already loaded |

Results include answers, calibrated logits, token encodings, inference time,
tokenization time, and actual computed token count for reproducible comparison.
Model loading time is printed to stderr. An invalid request exits with an error;
it is never silently truncated. The command-line runner is not an HTTP server.

The crate exposes `Runner::load(path, fused)` and `Runner::infer(request)` for the
recommended CPU path. `predict`, `predict_batched`, and their `prefix` argument retain
the individual comparison paths for experiments.

## Verify and benchmark

```bash
cargo test --release
.venv/bin/python scripts/validate.py
```

Validation runs the official **unmerged fp32 Python model** and the Rust full-row
and shared-prefix paths sequentially. It checks identical tokenization, option order,
raw probabilities, decisions, question isolation, and chunked cache continuation.
Independent kernel tests compare the fused CPU scan and depthwise convolution with
tensor implementations. Benchmark repeats exclude each request's first iteration.

The report is saved as `results/report.json`; individual parity summaries are retained
in `results/`. Full prediction JSONL files are local artifacts and ignored by Git.
These fixtures test implementation parity, not general model accuracy or calibration.

## Current scope

CPU and CUDA fp32 inference for the released Qwen3.5-based Kev-0.8B. Metal, quantization,
HTTP serving, and a cross-request prefix cache are not implemented. Long contexts use
chunked state prefill; the 64k admission limit does not establish 64k accuracy or latency.
Performance results apply only to the hardware and fixture set recorded in the report.

[RESEARCH.md](RESEARCH.md) records the architecture, sources, pins, comparison rule,
and optimization choices. [THIRD_PARTY.md](THIRD_PARTY.md) records source attribution.
