import sys

import torch

sys.path.insert(0, "src")
from model import LinkDegradationCNN  # noqa: E402


def test_forward_shape():
    m = LinkDegradationCNN()
    x = torch.randn(5, 3, 64)
    out = m(x)
    assert out.shape == (5,)


def test_weight_param_count_matches_manual_sum():
    m = LinkDegradationCNN()
    expected = (
        m.conv1.weight.numel() + m.conv2.weight.numel()
        + m.conv3.weight.numel() + m.fc.weight.numel()
    )
    assert m.weight_param_count() == expected
