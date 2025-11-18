"""
Export YOLOv8 model to ONNX, TensorRT, TorchScript for bottle-sorter.
"""
from bsort.models.yolov8 import YOLOv8Wrapper
from typing import Any
import os
import torch
from torch.onnx import export as onnx_export

try:
    import tensorrt as trt
    _HAS_TENSORRT = True
except ImportError:
    _HAS_TENSORRT = False


def export_model(model: YOLOv8Wrapper, export_dir: str) -> None:
    """Export model to ONNX, TensorRT, TorchScript."""
    os.makedirs(export_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(export_dir, "model.onnx")
    dummy_input = torch.randn(1, 3, model.img_size, model.img_size, device=model.device)
    onnx_export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX model exported to {onnx_path}")

    # Export to TorchScript
    torchscript_path = os.path.join(export_dir, "model.pt")
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(torchscript_path)
    print(f"TorchScript model exported to {torchscript_path}")

    # Export to TensorRT (if available)
    if _HAS_TENSORRT:
        trt_path = os.path.join(export_dir, "model.trt")
        with open(onnx_path, "rb") as f:
            onnx_model = f.read()
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse(onnx_model):
            print("Failed to parse ONNX model for TensorRT.")
            return
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        engine = builder.build_engine(network, config)
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())
        print(f"TensorRT engine exported to {trt_path}")

    # Validate exports
    validate_exports(model, onnx_path, torchscript_path)


def validate_exports(model: YOLOv8Wrapper, onnx_path: str, torchscript_path: str) -> None:
    """Validate exported models by comparing outputs."""
    dummy_input = torch.randn(1, 3, model.img_size, model.img_size, device=model.device)

    # Validate ONNX
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Validate TorchScript
    torchscript_model = torch.jit.load(torchscript_path)
    torchscript_outputs = torchscript_model(dummy_input)

    # Compare outputs
    original_outputs = model(dummy_input)
    assert torch.allclose(
        torch.tensor(ort_outputs[0]), original_outputs, atol=1e-4
    ), "ONNX outputs do not match original model."
    assert torch.allclose(
        torchscript_outputs, original_outputs, atol=1e-4
    ), "TorchScript outputs do not match original model."
    print("Exported models validated successfully.")
