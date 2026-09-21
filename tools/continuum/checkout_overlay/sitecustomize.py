"""Selected vLLM Python with shared native + build-installed FlashAttention.

vLLM's CMake installs FlashAttention Python helpers with the compiled kernels;
they are not completely present in source archives. Never fall back for the
vLLM scheduler/engine/model Python source.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys


if os.environ.get("BENCHMARK_SOURCE_ROOT"):
    root = Path(os.environ["BENCHMARK_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["BENCHMARK_RUNTIME_PACKAGE"]).resolve()

    class NativeOnlyFallback(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("vllm."):
                return None
            if fullname == "vllm.vllm_flash_attn":
                return importlib.util.spec_from_file_location(
                    fullname, runtime / "vllm_flash_attn/__init__.py")
            relative = Path(*fullname.split(".")[1:])
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                candidate = runtime / (str(relative) + suffix)
                if candidate.is_file():
                    return importlib.util.spec_from_file_location(fullname, candidate)
            return None

    # FlashAttention 必须双方共用完整构建依赖；其余只允许本地扩展回退。
    # 不回退到修改版的任何 Python 调度/引擎代码。
    sys.meta_path.insert(0, NativeOnlyFallback())
    import vllm
    if Path(vllm.__file__).resolve() != root / "vllm/__init__.py":
        # sitecustomize 普通异常会被解释器忽略；导入错源时必须直接拒绝运行。
        sys.stderr.write("FATAL: vLLM checkout origin mismatch\n")
        os._exit(78)
