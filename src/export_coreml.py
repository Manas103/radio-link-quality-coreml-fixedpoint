"""Export the trained float model to Core ML.

This machine is Windows, not macOS, so coremltools can convert the traced
PyTorch graph to a Core ML .mlmodel (the neuralnetwork backend; the newer
mlprogram/.mlpackage backend needs a native BlobWriter library that only
ships on macOS and is not available here, see README). What is NOT possible
on this machine is loading or running the exported .mlmodel: the Core ML
runtime itself is a macOS/iOS/visionOS framework. So this script proves the
export succeeds and the model's spec matches, but the float-vs-Core-ML
parity claim in the README is explicitly scoped to "the export is produced
from the same weights" rather than "the export was executed and diffed",
and is disclosed as such.
"""
import sys

import coremltools as ct
import torch

sys.path.insert(0, "src")
from model import LinkDegradationCNN  # noqa: E402


def main():
    model = LinkDegradationCNN()
    model.load_state_dict(torch.load("data/model.pt"))
    model.eval()

    example = torch.randn(1, 3, 64)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="kpi_window", shape=example.shape)],
        outputs=[ct.TensorType(name="degradation_logit")],
        convert_to="neuralnetwork",
    )
    out_path = r"coreml\link_degradation.mlmodel"
    mlmodel.save(out_path)

    spec = mlmodel.get_spec()
    n_layers = len(spec.neuralNetwork.layers)
    print(f"exported {out_path}, {n_layers} neural network layers in spec")
    with open("docs/coreml_export.txt", "w") as f:
        f.write(f"coremltools version: {ct.__version__}\n")
        f.write(f"convert_to: neuralnetwork (.mlmodel)\n")
        f.write(f"output path: {out_path}\n")
        f.write(f"spec layer count: {n_layers}\n")
        f.write(f"input: kpi_window {tuple(example.shape)}\n")
        f.write("Not executed on this machine: Core ML runtime is macOS/iOS-only;\n")
        f.write("this Windows machine proves export, not on-device execution.\n")


if __name__ == "__main__":
    main()
