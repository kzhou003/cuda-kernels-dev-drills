"""Build a drill's .cu file with torch.utils.cpp_extension.load_inline (G track, needs an NVIDIA GPU + nvcc).

The .cu file sits next to the Python wrapper (mine/ or solutions/). While the file still contains the
line `#define DRILL_STUB`, loading raises NotImplementedError: delete that line when you start.
The extension name includes a hash of the source, so editing the .cu file triggers a rebuild
(first build ~30-60 s, then cached in ~/.cache/torch_extensions).
"""
from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path


def _arch_flags() -> list[str]:
    import torch

    major, minor = torch.cuda.get_device_capability()
    # sm_120 (RTX 5090) / sm_90 (H100) / sm_100 (B200): build for the exact GPU in the box.
    # "a" suffix (sm_90a / sm_100a / sm_120a) unlocks arch-specific PTX (wgmma, tcgen05, block-scaled mma).
    cc = f"{major}{minor}"
    suffix = "a" if major >= 9 else ""
    return [f"-gencode=arch=compute_{cc}{suffix},code=sm_{cc}{suffix}"]


@functools.lru_cache(maxsize=None)
def _build(cu_path: str, src_hash: str, decls: str, functions: tuple[str, ...]):
    from torch.utils.cpp_extension import load_inline

    path = Path(cu_path)
    return load_inline(
        name=f"{path.parent.name}_{path.stem}_{src_hash}",
        cpp_sources=decls,
        cuda_sources=path.read_text(),
        functions=list(functions),
        extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "-std=c++17", *_arch_flags()],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("DRILL_VERBOSE") == "1",
    )


def load_drill(py_file: str, decls: str, functions: tuple[str, ...]):
    """Compile (or fetch from cache) the .cu file that sits next to `py_file`."""
    cu = Path(py_file).with_suffix(".cu")
    src = cu.read_text()
    if "#define DRILL_STUB" in src:
        raise NotImplementedError(f"{cu.parent.name}/{cu.name}: remove '#define DRILL_STUB' and implement the TODOs")
    return _build(str(cu), hashlib.sha1(src.encode()).hexdigest()[:10], decls, functions)
