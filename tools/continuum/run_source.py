#!/usr/bin/env python3
"""Run a module with this checkout's Python sources and installed vLLM binaries.

No install, symlink, editable-package or shell environment changes. The source
overlay is inherited by worker subprocesses through scoped PYTHONPATH.
"""
import importlib.util
# 中文导读：H200 旧环境的 editable 安装指向 workspaces 中的运行副本。
# 这里仅对子进程改导入路径：Python 用当前仓库，已编译 CUDA 扩展复用原环境。
# 不等于重新安装环境；版本不兼容时不能用这种方式混搭。
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    # This script's directory, not the repository root, is sys.path[0].
    import vllm
    runtime = Path(vllm.__file__).resolve().parent
    if importlib.util.find_spec("vllm._C") is None:
        raise SystemExit("Use the existing compiled vLLM environment's Python")
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_source.py MODULE [module arguments ...]")
    env = dict(os.environ)
    env["CONTINUUM_SOURCE_ROOT"] = str(root)
    # source_overlay/sitecustomize.py 会在解释器启动时读取这两个路径。
    env["CONTINUUM_RUNTIME_PACKAGE"] = str(runtime)
    overlay = Path(__file__).resolve().parent / "source_overlay"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(overlay), str(runtime.parent), env.get("PYTHONPATH", "")])
    print(f"Continuum Python source: {root}", flush=True)
    print(f"Compiled vLLM runtime: {runtime}", flush=True)
    return subprocess.call([sys.executable, "-m", *sys.argv[1:]],
                           cwd=root, env=env)


if __name__ == "__main__":
    sys.exit(main())
