# vLLM with Continuum Scheduling

The authors' public base (`316a587`) provided Continuum-style scheduling **without
the estimation in the paper**. This research checkout now independently implements
the paper's empirical-CDF TTL/cost model and pin lifecycle in the original V1
scheduler. See [Paper baseline implementation and verification](docs/continuum-paper.md)
for the exact assumptions, tests, calibration and source-checkout launcher.
Hardware/paper-performance reproduction is not yet claimed. Multi-node serving
requires sticky-session routing.

中文阅读入口：[新版源码结构、调用链、数据查看与小集验证](docs/continuum-guide.zh.md)。
核心模块已补中文注释；原始轨迹、小验证集和既有回放数据均保存在子仓库外，具体路径见中文导读。
2026-09-21：[arXiv v6 原文逐项核对](docs/continuum-conformance-v6.zh.md)，修正三级冷启动与被抢占请求优先级；不声称恢复了作者未公开的全部估计器细节。

You can use this as a base-repo to tune KV cache pin logic on different workloads.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Server](#starting-the-server)
  - [Original vLLM Mode](#original-vllm-mode)
  - [Continuum Scheduling Mode](#continuum-scheduling-mode)
- [Evaluation](#evaluation)
  - [Running SWE-bench Evaluation](#running-swe-bench-evaluation)
  - [Analyzing Results](#analyzing-results)

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager
- Hugging Face account with access token
- GPU(s) with appropriate CUDA drivers

## Installation

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode
uv pip install -e .

# Install mini-swe-agent
cd mini-swe-agent
uv pip install -e .
uv pip install datasets
cd ..

# Install additional dependencies
uv pip install lmcache hf_transfer

# Log in to Hugging Face (required for model access)
hf auth login
# Enter your Hugging Face access token when prompted
```

**Additional Setup**: Follow the instructions to set up [sb-cli](https://www.swebench.com/sb-cli/), which is required for pass rate evaluation.

## Usage

### Starting the Server

#### Original vLLM Mode

Run vLLM with standard scheduling:

```bash
# Without CPU offload
vllm serve <MODEL_NAME> \
  --tensor-parallel-size <NUM_GPUS> \
  --port <PORT_ID>

# With CPU offload (requires lmcache)
LMCACHE_MAX_LOCAL_CPU_SIZE=<CPU_SIZE_GB> \
vllm serve <MODEL_NAME> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --tensor-parallel-size <NUM_GPUS> \
  --port <PORT_ID>
```

#### Continuum Scheduling Mode

Paper Continuum requires a measured profile. The arXiv v6 cold-start prior is
derived from that cost model; it is not an arbitrary constant-TTL fallback.
The commands below assume this checkout is installed in your chosen environment.
On the existing H200-1 environment, use the source launcher documented above.

```bash
# Without CPU offload
vllm serve <MODEL_NAME> \
  --scheduling-policy continuum \
  --enable-prefix-caching \
  --additional-config '{"continuum":{"profile_path":"<PREFILL_PROFILE_JSON>"}}' \
  --tensor-parallel-size <NUM_GPUS> \
  --port <PORT_ID>

# With CPU offload (requires lmcache)
LMCACHE_MAX_LOCAL_CPU_SIZE=<CPU_SIZE_GB> \
vllm serve <MODEL_NAME> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --scheduling-policy continuum \
  --enable-prefix-caching \
  --additional-config '{"continuum":{"profile_path":"<RELOAD_PROFILE_JSON>"}}' \
  --tensor-parallel-size <NUM_GPUS> \
  --port <PORT_ID>
```

**Example:**
```bash
# Run Llama-3.1-70B-Instruct with Continuum on 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --scheduling-policy continuum \
  --enable-prefix-caching \
  --additional-config '{"continuum":{"profile_path":"<MATCHING_70B_TP4_PROFILE_JSON>"}}' \
  --tensor-parallel-size 4
```

## Evaluation

### Running SWE-bench Evaluation

**Note:** The default evaluation setup uses `meta-llama/Llama-3.1-70B-Instruct` on 4 H100 GPUs. Mini-swe-agent may encounter issues with smaller or simpler models.

1. **Start the vLLM server** (see [Usage](#usage) section above)

2. **Run the SWE-bench evaluation:**

```bash
# Clear previous output before each run
rm -rf ./swebench_output

# Run evaluation (fixed concurrency)
mini-extra swebench \
  --model-class vllm \
  --model <MODEL_NAME> \
  --port <PORT_ID> \
  --subset verified \
  --split test \
  --workers 64 \
  --output ./swebench_output

# Run evaluation (Poisson arrival rate)
mini-extra swebench \
  --model-class vllm \
  --model <MODEL_NAME> \
  --port <PORT_ID> \
  --subset verified \
  --split test \
  --use-jps --jps 1.0 \
  --output ./swebench_output
```

#### Load Control Modes

| Mode | Flag | Description |
|------|------|-------------|
| Workers | `--workers N` | Fixed N concurrent jobs |
| JPS | `--use-jps --jps X` | Poisson process at X jobs/second |

### Analyzing Results

**Important:** Terminate the vLLM server (Ctrl+C) before running the evaluation analysis.

```bash
# Analyze latency metrics
python continuum_exp/analyze.py \
  --output-dir <OUTPUT_DIRECTORY>

# Submit pass rate evaluation (use a unique run_id for each evaluation)
sb-cli submit swe-bench_verified test \
  --predictions_path swebench_output/preds.json \
  --run_id <UNIQUE_RUN_ID>
```

## Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `<MODEL_NAME>` | Hugging Face model identifier | `meta-llama/Llama-3.1-70B-Instruct` |
| `<NUM_GPUS>` | Number of GPUs for tensor parallelism | `4` |
| `<CPU_SIZE_GB>` | CPU memory size in GB for KV cache offload | `200` |
| `<OUTPUT_DIRECTORY>` | Directory for analysis output | `./continuum_exp/result` |
| `<UNIQUE_RUN_ID>` | Identifier for evaluation run | `continuum_run_001` |
| `--workers` | Fixed number of concurrent jobs | `64` |
| `--use-jps` | Enable Poisson arrival mode | - |
| `--jps` | Jobs per second (with `--use-jps`) | `1.0` |
