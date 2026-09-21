"""Run a selected pristine/modified checkout with shared compiled extensions.

Only child-process paths are changed. No package install or source rewrites.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("module")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import vllm
    runtime = Path(vllm.__file__).resolve().parent
    if importlib.util.find_spec("vllm._C") is None:
        parser.error("Use the existing compiled vLLM environment")
    root = args.source_root.resolve()
    if not (root / "vllm/__init__.py").is_file():
        parser.error("source-root is not a vLLM checkout")
    helper = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop("CONTINUUM_SOURCE_ROOT", None)
    env["BENCHMARK_SOURCE_ROOT"] = str(root)
    env["BENCHMARK_RUNTIME_PACKAGE"] = str(runtime)
    env["PYTHONPATH"] = os.pathsep.join(map(str, [
        helper / "tools/continuum/checkout_overlay", root, helper,
        helper / "mini-swe-agent/src"]))
    print(f"Selected Python checkout: {root}; compiled extensions: {runtime}", flush=True)
    # exec 保留同一个进程组，保护器只需管理本次启动的进程组。
    os.chdir(root)
    os.execvpe(sys.executable, [sys.executable, "-m", args.module, *args.arguments], env)


if __name__ == "__main__":
    main()
