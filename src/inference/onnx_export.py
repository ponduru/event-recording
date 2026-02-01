"""Export a PyTorch checkpoint to ONNX format for lightweight inference."""

import argparse
from pathlib import Path

import torch

from src.core.domain import DomainRegistry
import src.domains  # noqa: F401 - trigger domain registration


def export_to_onnx(
    checkpoint_path: str,
    output_path: str | None = None,
    window_size: int = 8,
    frame_size: tuple[int, int] = (224, 224),
    opset_version: int = 17,
) -> Path:
    """Export a PyTorch checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        output_path: Path for .onnx output. Defaults to same dir/name with .onnx extension.
        window_size: Number of frames per window (must match inference).
        frame_size: (width, height) of input frames.
        opset_version: ONNX opset version.

    Returns:
        Path to exported ONNX model.
    """
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = Path(output_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    domain_name = checkpoint.get("domain", "cricket")
    model_config = checkpoint.get("config", {})

    domain = DomainRegistry.get(domain_name)
    model = domain.create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy input: (batch=1, T=window_size, C=3, H, W)
    h, w = frame_size[1], frame_size[0]
    dummy_input = torch.randn(1, window_size, 3, h, w)

    # Export using legacy ONNX exporter (doesn't require onnxscript)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={
            "frames": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"  Domain: {domain_name}")
    print(f"  Input shape: (batch, {window_size}, 3, {h}, {w})")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("-o", "--output", help="Output .onnx path (default: same name)")
    parser.add_argument("--window-size", type=int, default=8, help="Frames per window")
    parser.add_argument("--frame-width", type=int, default=224, help="Frame width")
    parser.add_argument("--frame-height", type=int, default=224, help="Frame height")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(
        args.checkpoint,
        args.output,
        window_size=args.window_size,
        frame_size=(args.frame_width, args.frame_height),
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
