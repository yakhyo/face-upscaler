import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from models import SPAN


def load_model(model_path, device):
    """
    Loads the SPAN model and its weights.

    Args:
        model_path (str): Path to the model weights.
        device (torch.device): Compute device (CPU or GPU).

    Returns:
        torch.nn.Module: Loaded and ready-to-use model.
    """
    model = SPAN(num_in_ch=3, num_out_ch=3)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)["params"]
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def preprocess(image_path, device):
    """
    Preprocesses the input image.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Compute device.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)  # Add batch dimension and send to device


def postprocess(output_tensor):
    """
    Post-processes the output tensor into a NumPy image array.

    Args:
        output_tensor (torch.Tensor): Model output tensor.

    Returns:
        np.ndarray: Processed image in HWC format (RGB).
    """
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp to [0, 1]
    output_image = output_tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    output_image = (output_image * 255).astype("uint8")  # Scale to [0, 255]
    return output_image


def save_output(output_image, save_dir, image_path):
    """
    Saves the processed image to the specified directory.

    Args:
        output_image (np.ndarray): Processed image in HWC format (RGB).
        save_dir (str): Directory to save the output image.
        image_path (str): Original input image path (used for naming the output file).
    """
    # Extract file name and add '_out' postfix
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))
    output_file_name = f"{file_name}_out{file_ext}"
    output_path = os.path.join(save_dir, output_file_name)

    # Convert RGB to BGR (OpenCV uses BGR format)
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Save the image using OpenCV
    cv2.imwrite(output_path, output_image_bgr)
    print(f"Super-resolved image saved at {output_path}")


def run_inference(model_path, image_path, save_dir):
    """
    Runs the super-resolution inference pipeline.

    Args:
        model_path (str): Path to the model weights.
        image_path (str): Path to the input image.
        save_dir (str): Directory to save the super-resolved image.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)

    # Preprocess image
    input_tensor = preprocess(image_path, device)

    # Perform inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Post-process the output tensor
    output_image = postprocess(output_tensor)

    # Save the output image
    save_output(output_image, save_dir, image_path)


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-resolution inference pipeline.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/4x-ClearRealityV1.pth",
        help="Path to the model weights. Default: weights/4x-ClearRealityV1.pth"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="assets/image0.jpg",
        help="Path to the input image file. Default: assets/image0.jpg"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output",
        help="Directory to save the super-resolved image. Default: output"
    )
    return parser.parse_args()


def main():
    """
    Main function to run the super-resolution inference pipeline.
    """
    # Parse arguments
    args = parse_arguments()

    # Run the pipeline
    run_inference(args.model_path, args.image_path, args.save_dir)


if __name__ == "__main__":
    main()
