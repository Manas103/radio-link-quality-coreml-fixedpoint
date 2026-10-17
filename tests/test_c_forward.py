"""Compiles and runs the C fixed-point forward pass against the tiny fixture
weights/windows checked in under tests/fixtures, and checks it produces a
finite probability in [0, 1] for every window. This is a correctness smoke
test, not the parity measurement itself (that is measured once, on the real
2,000-window held-out set, and recorded in docs/ and build_report.json).
"""
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _msvc_available():
    return shutil.which("cl") is not None


@pytest.mark.skipif(not (ROOT / "c" / "weights.h").exists(), reason="run src/quantize.py first")
def test_c_binary_builds_and_runs_on_real_parity_windows(tmp_path):
    cl = shutil.which("cl")
    if cl is None:
        pytest.skip("MSVC cl.exe not on PATH in this shell; built via vcvars64.bat in CI/build steps")
    exe = tmp_path / "forward.exe"
    r = subprocess.run(
        ["cl", "/nologo", "/O2", "/std:c11", f"/Fe:{exe}", str(ROOT / "c" / "forward.c")],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    windows_path = ROOT / "data" / "parity_windows.f32"
    ref_path = ROOT / "data" / "parity_reference_probs.f32"
    if not windows_path.exists():
        pytest.skip("data/parity_windows.f32 not generated; run src/quantize.py first")
    r = subprocess.run([str(exe), "parity", str(windows_path), str(ref_path)],
                        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "max_abs_diff=" in r.stdout

    out = ROOT / "docs" / "parity_c_probs.f32"
    data = out.read_bytes()
    n = len(data) // 4
    probs = struct.unpack(f"<{n}f", data)
    assert n > 0
    assert all(0.0 <= p <= 1.0 for p in probs)
