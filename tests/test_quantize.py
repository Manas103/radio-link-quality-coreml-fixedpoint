import sys

import numpy as np

sys.path.insert(0, "src")
from quantize import decompose_multiplier, quantize_per_out_channel  # noqa: E402


def test_decompose_multiplier_reconstructs_within_tolerance():
    for real_multiplier in (0.5, 0.01, 0.9999, 0.0001, 1.0, 2.3):
        m, shift = decompose_multiplier(real_multiplier)
        approx = m / (2.0 ** shift)
        assert abs(approx - real_multiplier) / real_multiplier < 1e-5


def test_quantize_per_out_channel_bounds_and_scale():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, size=(4, 3, 5)).astype(np.float32)
    w[1] *= 10  # one channel with a much larger range than the others
    q, scales = quantize_per_out_channel(w)
    assert q.dtype == np.int32
    assert q.min() >= -128 and q.max() <= 127
    assert scales.shape == (4,)
    # the per-channel scale for the large-range channel must be larger,
    # which is the whole point of quantizing per channel instead of per tensor
    assert scales[1] > scales[0]


def test_requantize_math_matches_c_int64_scheme():
    """Mirrors forward.c's requantize(): (acc * M + round) >> shift, all integer."""
    acc = 123456
    real_multiplier = 0.0031415
    m, shift = decompose_multiplier(real_multiplier)
    c_style = (acc * m + (1 << (shift - 1))) >> shift
    float_ref = acc * real_multiplier
    assert abs(c_style - float_ref) <= 1  # within one quantization step
