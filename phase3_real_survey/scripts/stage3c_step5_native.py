"""Build and call the optional Stage 3C C++ timing-bootstrap kernel."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "stage3c_bootstrap_timing.cpp"
BUILD_DIR = ROOT / ".native_build"
LIBRARY = BUILD_DIR / (
    "libstage3c_bootstrap_timing.dylib"
    if platform.system() == "Darwin"
    else "libstage3c_bootstrap_timing.so"
)
METRIC_NAMES = (
    "signed_median_days",
    "median_absolute_error_days",
    "mad_days",
    "iqr_days",
    "q10_days",
    "q90_days",
)

_LIBRARY_HANDLE = None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiler():
    requested = os.environ.get("CXX")
    if requested:
        resolved = shutil.which(requested)
        if resolved:
            return resolved
        raise RuntimeError(f"CXX compiler not found: {requested}")
    for candidate in ("clang++", "g++", "c++"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("No C++17 compiler was found.")


def build_native_library(force=False):
    """Compile the local kernel when absent or older than its source."""
    if not SOURCE.is_file():
        raise RuntimeError(f"Native source is missing: {SOURCE}")
    if (
        not force
        and LIBRARY.is_file()
        and LIBRARY.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns
    ):
        return LIBRARY

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = LIBRARY.with_suffix(LIBRARY.suffix + ".tmp")
    command = [_compiler(), "-O3", "-std=c++17"]
    if platform.system() == "Darwin":
        command.extend(["-dynamiclib", str(SOURCE), "-o", str(temporary)])
    else:
        command.extend([
            "-shared", "-fPIC", str(SOURCE), "-o", str(temporary)
        ])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Native timing kernel compilation failed:\n"
            f"command: {' '.join(command)}\n{completed.stderr}"
        )
    os.replace(temporary, LIBRARY)
    return LIBRARY


def _load_library():
    global _LIBRARY_HANDLE
    if _LIBRARY_HANDLE is None:
        library = ctypes.CDLL(str(build_native_library()))
        function = library.stage3c_bootstrap_timing
        function.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
        _LIBRARY_HANDLE = library
    return _LIBRARY_HANDLE


def bootstrap_timing_statistics_native(values, available, indices):
    """Return frozen timing bootstrap summaries from the C++17 kernel."""
    x = np.ascontiguousarray(values, dtype=np.float64)
    mask = np.ascontiguousarray(available, dtype=np.uint8)
    idx = np.ascontiguousarray(indices, dtype=np.int64)

    if x.ndim != 1 or mask.shape != x.shape:
        raise ValueError("Values and availability mask must be aligned.")
    if idx.ndim != 2 or idx.shape[1] != x.size:
        raise ValueError("Bootstrap indices have incompatible dimensions.")
    if np.any(mask.astype(bool) & ~np.isfinite(x)):
        raise ValueError("Available timing errors must be finite.")

    output = np.empty((idx.shape[0], len(METRIC_NAMES)), dtype=np.float64)
    function = _load_library().stage3c_bootstrap_timing
    status = function(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        idx.shape[0],
        x.size,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"Native timing kernel failed with status {status}.")
    return {
        name: output[:, column].copy()
        for column, name in enumerate(METRIC_NAMES)
    }


def native_backend_provenance():
    """Describe the compiled implementation used for reproducibility."""
    compiler = _compiler()
    library = build_native_library()
    version = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "compiler_path": compiler,
        "compiler_version": version,
        "compile_flags": ["-O3", "-std=c++17"],
        "source_path": str(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "library_path": str(library),
        "library_sha256": _sha256(library),
    }
