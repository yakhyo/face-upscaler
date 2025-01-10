import torch
from models import SPAN


def export_to_onnx(model, onnx_path="model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        },
    )
    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    # Load the model
    model = SPAN(num_in_ch=3, num_out_ch=3)
    state_dict = torch.load("weights/4x-ClearRealityV1.pth", weights_only=True)["params"]
    model.load_state_dict(state_dict)

    # Export to ONNX
    export_to_onnx(model, "weights/4x-ClearRealityV1.onnx")
