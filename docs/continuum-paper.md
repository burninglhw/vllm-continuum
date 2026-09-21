# Paper Continuum baseline: implementation and verification

This checkout extends the authors' public commit
`316a58794a6ff86b216e579b74fd56ed0c5a911f` in place. The upstream README explicitly
described its implementation as **without the estimation in the paper**.
This change reconstructs that missing model from Sections 3.1-3.3, Algorithm 1
and Appendix C of *Continuum* (Lifelong Agent Workshop @ ICLR 2026). The
2026-09-21 audit additionally uses **arXiv:2511.02230v6 (2026-05-25)**,
especially its explicit cold-start rules (§4.2) and scheduling priorities (§4.3).
It is an independently reconstructed paper implementation, **not recovered
unpublished author code**, and unit-test success is not a reproduction of the
paper's performance numbers.

The earlier reconstruction omitted the v6 three-stage cold start and preempted
request priority. These are now corrected. See
[逐项原文核对与仍未公开的细节](continuum-conformance-v6.zh.md) for the evidence,
boundary tests and limitations. Matching the published equations does not prove
identity with the authors' unpublished estimator implementation.

No imports or execution paths use `elastic-kv-study`, `ElasticScheduler`, or
`continuum-public`. The entry is the original V1 `Scheduler` with
`--scheduling-policy continuum`. FCFS/priority do not initialize the tool handler.

## Implementation map

| File | Responsibility |
| --- | --- |
| `vllm/v1/core/continuum_policy.py` | Three-stage TTL optimizer (exponential prior / global CDF / per-tool CDF), Pearson eta, evicted-wait history, measured reconstruction-cost profile |
| `vllm/v1/core/estimate_with_func.py` | Server tool parsing, causal inter-turn timings, program completion history, decision logs |
| `vllm/v1/core/sched/scheduler.py` | Whole-request pin ownership, expiration, safe reference transfer, deadlock recovery and preemption |
| `vllm/v1/core/sched/request_queue.py` | Preempted first, then pinned among other requests, then program-level FCFS |
| `vllm/v1/request.py` | Server-derived pin deadline and terminal state |
| `mini-swe-agent/src/minisweagent/models/vllm_model.py` | Stable program ID, including default/zero-ID clients |
| `tools/continuum/profile_prefill.py` | Offline hardware/model calibration, quadratic fit and raw measurements |
| `tools/continuum/run_source.py` | Process-scoped source launcher using existing compiled dependencies |
| `tests/continuum/` | Formula, handler, client, real scheduler and real KV block-pool tests |

## Paper correspondence and explicit assumptions

The TTL is the maximizer of Equation (2). In the empirical stages:

```text
F_tool(tau) * (mean_evicted_queue_delay * eta + PrefillReload(context)) - tau
tau in {0} union unique observed durations
eta = -PearsonCorr(k, N-k)
```

- **Cold-start source selection (v6 §4.2):** K=100. If total observed tool
  returns |S|<=100, use the Exp(1 second) prior and eta=1. Otherwise, if
  |S[f]|<=100, pool every tool-call observation into the global empirical CDF;
  otherwise use the selected tool's empirical CDF. T is initialized to zero.
  These tests are inclusive at 100; the next stage begins at 101.
- Substituting the exponential prior in Equation (2), with B=T+PrefillReload(r)
  in seconds and mean mu=1 second, gives tau=mu*log(B/mu) when B>mu, otherwise
  zero. This is a mathematical derivation, not a paper-specified numeric
  constant. The paper calls this T_default but does not publish one universal
  numeric TTL or a representative context/refresh rule; we evaluate the stated
  objective with this request's cost and the currently observed T.
- Enumerate the empirical CDF, including repeated observations; ties prefer the
  smaller TTL. This is not a threshold on mean tool duration.
- Durations are the next request's **server arrival timestamp** minus the
  previous request's completion timestamp. Handler-processing/queue delays do
  not enter the tool duration; negative/nonfinite observations are discarded.
- No future total-step counts, trace-provided tool durations, or future runtime
  information enter the policy. `is_last_step` is read only after completion,
  never for scheduling/preemption priority.
- Full-context prefill reconstruction is used, even if some unpinned pages may
  survive opportunistically, as specified in Appendix C.2.
- Expired pins remain protected while their program has a queued return.
  Expiration uses the strict `now > deadline` predicate from Algorithm 1.
  After successful allocation acquires prefix references, release the old
  request's references. Expiration scans a snapshot, not a mutating iterator.
- Memory pressure releases whole pinned requests, latest **program arrival**
  first (not latest TTL expiry). Admission retries in the same step, refreshing
  cache lookup each time. No Elastic-style partial/suffix reclamation exists.
- If running decode cannot allocate, unpin before preempting running work;
  running victims use latest program arrival, with no last-step oracle.
- Cascade attention is disabled for Continuum: retained/delayed-free owners
  invalidate its assumption that block reference counts equal batch size.

The paper leaves these choices underspecified; these are our explicit defaults:

1. **Undefined empirical statistics:** eta is zero when empirical correlation
   is undefined and the queue average is zero before any sample. The separate
   v6 cold-start branch overrides eta to 1, as explicitly required by the paper.
   Unknown tools do not unconditionally get TTL=0; they use the prior or global
   history according to the thresholds above.
2. **Windows:** queue delay uses the last 256 evicted waiting episodes; eta uses
   the last 256 completed programs. Each completed N-step program contributes
   `(k, N-k)` for every non-final boundary `k=1,...,N-1`. Samples are pooled across
   completed programs; incomplete programs never reveal N. These sample/window
   conventions are not claimed to be the authors' unpublished choices.
3. **Correlation:** zero when undefined; otherwise retain its signed value in
   `[-1,1]`, rather than replacing it with a regression slope or clipping to a
   memoryfulness prior.
4. **Eviction measurement:** an unpinned/TTL-zero program's next request measures
   wait from arrival to admission; an active preempted request measures from
   preemption to readmission. Fresh non-evicted requests do not train this cost.
5. **Completion:** a normal completion without a recognized tool call, an
   explicit terminal flag, or mini-SWE's final-output echo ends the program.
   Abort, length cap and parser exceptions free state without training program
   length. Unsupported output formats can look terminal; adapt/test the parser
   before using a different agent protocol.

## Supported scope and limitations

The initial target is sequential, text-only agent programs on V1 with prefix
caching and synchronous scheduling. IDs must be unique among simultaneous
programs and stable across turns; overlapping requests for the same program ID
are explicitly rejected. With no ID, requests stay isolated and are not pinned.
Use sticky routing if running multiple independent server/engine instances.
Abandoned programs release pins on subsequent scheduler steps after TTL, but
their small pending tool-history records remain until a return/terminal event;
use bounded experiment processes for now.

The parser supports a single bash block (Appendix D), terminal mini-SWE echo,
and a single JSON function invocation (plain, `tool_calls`, JSON fence or
`<tool_call>` wrapper). Arbitrary model-specific reasoning/tool formats,
parallel/async tools, multimodal cost models and general P/D disaggregation
are not validated. Async scheduling is rejected, not silently enabled.
Use EOS-based completion for the first baseline. Frontend stop-string clipping
is delivered to this V1 engine as an abort and conservatively will not pin;
distinguishing it from client cancellation is not implemented here.

CPU offload has a bandwidth-based cost path and reference-safe delayed-free
handling, but **LMCache transfer correctness and hardware throughput have not
been validated by the CPU-only tests**. A reload profile does not itself enable
offload. No-offload prefill is the first baseline to calibrate and benchmark.

## Run the changed checkout without reinstalling the existing environment

On H200-1, the installed editable package still points at the old runtime copy.
Running a bare `vllm serve` is therefore not a reliable way to test this checkout.
The following wrapper changes import paths only for its child process/workers,
using Python source from this repository and compiled extensions/dependencies
from the installed runtime. It also selects this checkout's mini-SWE source.
It does not install, overwrite or relink anything in the environment.

```bash
cd /export/home/ext.luohaowen1/continuum/vllm-continuum
/export/home/ext.luohaowen1/continuum/envs/continuum/bin/python \
  tools/continuum/run_source.py unittest discover -s tests/continuum -v
```

Do not use this binary-overlay technique across incompatible vLLM/kernel
versions. The existing runtime is the separately prepared 316a587-compatible
environment; pin its dependency manifest along with any benchmark results.

## Calibrate before real serving

Reserve an idle GPU first. Do not profile alongside another experiment: queue
and interference artifacts invalidate the cost model. Replace placeholders;
no particular GPU index or model is selected automatically by this document.

```bash
CUDA_VISIBLE_DEVICES=<RESERVED_GPU> \
/export/home/ext.luohaowen1/continuum/envs/continuum/bin/python \
  tools/continuum/run_source.py tools.continuum.profile_prefill \
  --model <LOCAL_MODEL_PATH> --max-context 16384 --repeats 5 \
  --gpu-memory-utilization 0.5 --enforce-eager \
  --output /export/home/ext.luohaowen1/continuum/reproduction/continuum-paper/prefill-profile.json
```

The profiler disables prefix caching, warms up each size and measures isolated
prefills at `{1000,2000,4000,...,max_context-1}`. The final point reserves one
output token under vLLM's context limit; the fitted model covers that last token
by one-token extrapolation. It reads V1's `request_prefill_time_seconds`
histogram, **not RequestOutput.metrics (unset in this V1 version)**. This is the
engine scheduled-to-first-token interval, including one-token sampling overhead
but excluding queue time. It saves every sample, quadratic coefficients
`[a,b,c]` in seconds, fit RMSE, model/dtype/TP/eager settings and GPU name.
Review the curve and residuals before accepting a calibration.

Now launch the calibrated baseline (same model, dtype, TP, GPU hardware and
execution settings as profiling):

```bash
CUDA_VISIBLE_DEVICES=<RESERVED_GPU> \
RUN_OUTPUT_DIR=/export/home/ext.luohaowen1/continuum/reproduction/continuum-paper/run-001 \
/export/home/ext.luohaowen1/continuum/envs/continuum/bin/python \
  tools/continuum/run_source.py vllm.entrypoints.openai.api_server \
  --model <LOCAL_MODEL_PATH> --scheduling-policy continuum \
  --enable-prefix-caching --max-model-len 16384 --enforce-eager \
  --gpu-memory-utilization 0.5 --port <UNUSED_PORT> \
  --additional-config '{"continuum":{"profile_path":"/export/home/ext.luohaowen1/continuum/reproduction/continuum-paper/prefill-profile.json","queue_window":256,"program_window":256}}'
```

The model path must match the profile. Model identity, covered context length,
and provided dtype/TP/eager metadata are checked. GPU identity is recorded for
the experimenter to check (the CPU scheduler does not initialize a CUDA device
to validate it). Missing profiles and incompatible offload modes fail at startup.
For a fair FCFS comparison use the same command/resources and change the policy
to `fcfs`; the Continuum configuration is not consumed in that mode.

Clients send `extra_body={"job_id":"unique-program-id"}` on every turn and use
the same ID through the program. Do not supply future steps/tool runtimes from
an evaluation trace to the policy. Terminal flags may only describe the current
program's actual stopping condition.

### Reload profile schema (experimental, requires actual measurements)

```json
{
  "schema_version": 1,
  "mode": "reload",
  "model": "<exact server model path>",
  "max_context": 16384,
  "bytes_per_token": "REPLACE_WITH_MEASURED_KV_BYTES_PER_TOKEN",
  "bytes_per_second": "REPLACE_WITH_MEASURED_EFFECTIVE_CPU_GPU_THROUGHPUT"
}
```

Replace strings with measured positive numbers. Match per-rank or aggregate
bytes to the same bandwidth convention; do not use a device's advertised peak
bandwidth. The connector must actually preserve/reload the data. Benchmark
prefill fallback separately when CPU storage misses. SSD offload is not claimed
validated here.

## Audit and reproducibility

On graceful server shutdown, `RUN_OUTPUT_DIR/scheduler_timestamps` includes
per-program arrivals, admissions/cache hits, pin/unpin events and TTL decisions
with CDF sample count, eta, average evicted wait and reconstruction seconds.
Each decision also records `estimation_source`, `cdf_samples`,
`total_tool_samples`, `empirical_eta` and `cold_start_threshold`. The existing
`eta` field is the coefficient actually used, including eta=1 during cold start.
Keep the source diff/commit, profile, model revision, environment manifest,
dataset split, random seed, warm-up policy and run settings together.

The v6 conformance audit runs deterministic CPU-side tests using real
scheduler/block managers. It does not run GPU inference or profiling, LMCache,
SWE-bench quality scoring or load sweeps, and makes no paper-speedup claim.
Previous shared-H200 calibration is preliminary, not final performance evidence.
Code changes are left uncommitted; no GitHub push or parent submodule gitlink
update is performed automatically.
