"""Process-scoped source overlay, activated only by run_source.py."""
import os
import sys
from pathlib import Path

if os.environ.get("CONTINUUM_SOURCE_ROOT"):
    import vllm
    source = str(Path(os.environ["CONTINUUM_SOURCE_ROOT"]) / "vllm")
    runtime = os.environ["CONTINUUM_RUNTIME_PACKAGE"]
    vllm.__path__[:] = list(dict.fromkeys([source, runtime, *vllm.__path__]))
    sys.path.insert(0, str(Path(os.environ["CONTINUUM_SOURCE_ROOT"]) /
                           "mini-swe-agent/src"))
