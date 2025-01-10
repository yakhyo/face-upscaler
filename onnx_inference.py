import os
import cv2
import argparse
import numpy as np
from PIL import Image

import onnxruntime as ort

import torchvision.transforms as transforms


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run ONNX model inference on an input image.")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="weights/4x-ClearRealityV1.onnx",
        help="Path to the ONNX model file. Default: weights/4x-ClearRealityV1.onnx"
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
        help="Directory to save the output image. Default: output"
    )
    return parser.parse_args()


class ClearRealityV1:
    def __init__(self, onnx_path: str, input_size: int = 128) -> None:
        """
        Initializes the ONNX runtime session.

        Args:
            onnx_path (str): Path to the ONNX model file.
            input_size (int): Input image size expected by the model.
        """
        assert input_size in [128, 256, 512], "Only input sizes of 128, 256 and 512 are supported."
        self.input_size = input_size

        self.ort_session = ort.InferenceSession(onnx_path)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input NumPy image to match the ONNX model's expected input format.

        Args:
            image (np.ndarray): Input image as a NumPy array in HWC format (BGR).

        Returns:
            np.ndarray: Preprocessed image tensor in NCHW format.
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Define preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

        # Apply transforms
        tensor_image = transform(pil_image)

        # Convert to NumPy array and add batch dimension (NCHW)
        numpy_image = tensor_image.unsqueeze(0).numpy()

        return numpy_image

    @staticmethod
    def postprocess(model_output):
        """
        Postprocesses the output tensor into an image.

        Args:
            model_output (np.ndarray): Model output tensor in NCHW format.

        Returns:
            np.ndarray: Processed image in HWC format (RGB) as a NumPy array.
        """
        # Remove batch dimension and clamp values to [0, 1]
        model_output = np.clip(model_output.squeeze(0), 0, 1)

        # Convert CHW to HWC
        output_image = np.transpose(model_output, (1, 2, 0))

        # Scale to [0, 255] and convert to uint8
        output_image = (output_image * 255).astype(np.uint8)

        return output_image

    def run_inference(self, image):
        """
        Runs the inference pipeline.

        Args:
            image (np.ndarray): Input image as a NumPy array in HWC format (BGR).

        Returns:
            np.ndarray: Model output as a processed image in HWC format (RGB).
        """
        # Preprocess the input image
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.ort_session.run(None, {"input": input_tensor})

        # Postprocess the output tensor into an image
        return self.postprocess(outputs[0])


def save_output(processed_image, save_dir, image_path):
    """
    Saves the processed image to the specified directory.

    Args:
        processed_image (np.ndarray): Processed image in HWC format (RGB).
        save_dir (str): Directory to save the output image.
        image_path (str): Path to the input image, used for naming the output file.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Prepare the output file name
    input_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_out{ext}"
    output_path = os.path.join(save_dir, output_filename)

    # Convert the processed image to BGR for OpenCV compatibility
    processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite(output_path, processed_image_bgr)
    print(f"Super-resolved image saved at {output_path}")


def main(params):
    """
    Main function to run the ONNX inference pipeline.
    """
    # Load the input image using OpenCV
    input_image = cv2.imread(params.image_path)  # BGR format from OpenCV
    if input_image is None:
        raise ValueError(f"Unable to read the input image at {params.image_path}")

    # Initialize ONNXInference
    inference = ClearRealityV1(params.onnx_path)

    # Run inference
    processed_image = inference.run_inference(input_image)

    # Save the output image
    save_output(processed_image, params.save_dir, params.image_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
