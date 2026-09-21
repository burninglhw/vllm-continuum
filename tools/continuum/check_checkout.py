"""CPU-only origin/extension compatibility check before loading model weights."""
import hashlib
import importlib
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ["BENCHMARK_SOURCE_ROOT"])
    result = {}
    for name in ("vllm", "vllm.v1.core.sched.scheduler", "vllm.v1.core.kv_cache_manager",
                 "vllm.v1.engine.core", "vllm.v1.request", "vllm.model_executor.models.llama", "vllm._C",
                 "vllm.vllm_flash_attn.layers.rotary", "vllm.vllm_flash_attn.ops.triton.rotary",
                 "vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path.suffix == ".py" and not name.startswith("vllm.vllm_flash_attn."):
            assert path.is_relative_to(root), (name, path)
        result[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
