"""Export a trained checkpoint to ONNX for the browser demo (docs/index.html).

    python scripts/export_onnx.py outputs/cnn_sqrtw/best.pt docs/fer_cnn.onnx

Verifies the exported model against PyTorch with onnxruntime.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import build_model  # noqa: E402


def main(ckpt_path, out_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = build_model(c["model"], 7, pretrained=False)
    m.load_state_dict(c["state_dict"])
    m.eval()
    torch.onnx.export(
        m, torch.randn(1, 1, 48, 48), out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    s = ort.InferenceSession(out_path)
    x = torch.randn(8, 1, 48, 48)
    diff = np.abs(s.run(None, {"input": x.numpy()})[0] - m(x).detach().numpy()).max()
    print(f"Saved {out_path}; max abs difference vs PyTorch: {diff:.2e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
