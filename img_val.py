import cv2  # For image processing
import numpy as np  # For array operations
from PIL import Image  # For image processing
import OpenEXR  # For opening EXR files
import Imath  # For handling EXR files
import os  # For handling file paths
from utils.logger_config import configure_logger  # For logging configuration
import json  # For handling JSON files

# Load color chart values from a JSON file
with open('color_charts_values/color_chart_values.json', 'r') as file:
    color_charts = json.load(file)

rgb_color_chart_values = color_charts['RGB_color_chart_values']
aces2065_1_color_chart_values = color_charts['aces2065_1_color_chart_values']
itu_r_bt_2020_color_chart_values = color_charts['itu_r_bt_2020_color_chart_values']


def load_image_rgb(image_path):
    """
    Load an RGB image using PIL and print info messages.
    """
    global logger
    logger = configure_logger('rgb')
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")

    filename = os.path.basename(image_path)
    logger.info("RGB Color Check")
    logger.info("Loading image, please wait...")

    try:
        image = Image.open(image_path)
        image = np.array(image)  # convert PIL Image to numpy array
    except Exception as image_load_exception:
        logger.error(f"Failed to load image: {image_load_exception}")
        raise

    logger.info(f"Successfully loaded {filename}")
    logger.info(f"Color checking starting...")
    return image


def load_image_aces(image_path):
    """Load an ACES image file and return it as a numpy array."""
    global logger
    logger = configure_logger('aces2065_1')
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")

    filename = os.path.basename(image_path)
    logger.info("ACES2065_1 Color Check")
    logger.info("Loading image...")

    if image_path.lower().endswith(".exr"):
        exr_file = OpenEXR.InputFile(image_path)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r_str, g_str, b_str = exr_file.channel('R', pt), exr_file.channel('G', pt), exr_file.channel('B', pt)

        r_channel, g_channel, b_channel = [np.frombuffer(channel_str, dtype=np.float32).reshape((size[1], size[0])) for
                                           channel_str in [r_str, g_str, b_str]]
        image = np.dstack((r_channel, g_channel, b_channel))
        logger.info(f"Successfully loaded {filename}")
    else:
        logger.error(f"Loading non-EXR file for ACES: {image_path}")
        raise ValueError(f"Non-EXR file provided for ACES color chart: {image_path}")

    logger.info("Color checking starting...")
    return image


def load_image_itu2020(image_path):
    """
    Load an ITU2020 image using OpenEXR and PIL, and print info messages.
    """
    global logger
    logger = configure_logger('itu2020')
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")

    filename = os.path.basename(image_path)
    logger.info("ITU_BT2020 Color Check")
    logger.info("Loading image...")

    if image_path.lower().endswith(".exr"):
        exr_file = OpenEXR.InputFile(image_path)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r_str, g_str, b_str = exr_file.channel('R', pt), exr_file.channel('G', pt), exr_file.channel('B', pt)

        r_channel, g_channel, b_channel = [np.frombuffer(channel_str, dtype=np.float32).reshape((size[1], size[0])) for
                                           channel_str in [r_str, g_str, b_str]]
        image = np.dstack((r_channel, g_channel, b_channel))
        logger.info(f"Successfully loaded {filename}")
    else:
        logger.info(f"Loading non-EXR file: {image_path}")
        image = np.array(Image.open(image_path)) / 255.0

    logger.info("Color checking starting...")
    return image


def validate_color_chart(image, chart_values, tolerance=0.05):
    """Check the color accuracy of an image based on a color chart."""
    passed, failed, total_accuracy = 0, 0, 0

    for value in chart_values:
        if not all(key in value for key in ["rgb", "x", "y"]):
            raise ValueError("Each color patch dictionary must have 'rgb', 'x', and 'y' keys.")

        expected_rgb = np.array(value["rgb"])
        x, y = value["x"], value["y"]
        actual_rgb = image[y, x]

        # Calculate the Euclidean distance between the expected and actual RGB values
        distance = np.linalg.norm(expected_rgb - actual_rgb)

        # The maximum possible distance is the distance from black (0, 0, 0) to white (255, 255, 255)
        max_distance = np.linalg.norm([255, 255, 255])

        # Calculate accuracy as 1 - (distance / max_distance)
        accuracy = 1 - (distance / max_distance)

        value["accuracy"] = accuracy
        if accuracy >= (1 - tolerance):
            passed += 1
            total_accuracy += accuracy
        else:
            failed += 1

        logger.info(f"Color name: {value.get('name', 'Unknown')}")
        logger.info(f"Expected RGB: {expected_rgb}")
        logger.info(f"Actual RGB: {actual_rgb}")
        logger.info(f"Accuracy: {accuracy:.2%}")

    overall_accuracy = total_accuracy / len(chart_values)
    logger.info(f"Passed checks: {passed}")
    logger.info(f"Failed checks: {failed}")
    logger.info(f"Overall accuracy: {overall_accuracy:.2%}")

    return passed, failed, overall_accuracy, chart_values


def configure_loggers(color_space):
    # Initialize logger for the specified color space
    loggers = {
        'rgb': configure_logger('rgb', 'rgb.log'),
        'aces2065_1': configure_logger('aces2065_1', 'aces2065_1.log'),
        'itu2020': configure_logger('itu2020', 'itu2020.log')
    }
    return loggers[color_space]


def validate_single_image(image_path, color_space):
    # Fetch the appropriate logger
    global logger
    logger = configure_logger(color_space)

    if color_space == "rgb":
        image = load_image_rgb(image_path)
        color_chart_values = rgb_color_chart_values
    elif color_space == "aces2065_1":
        image = load_image_aces(image_path)
        color_chart_values = aces2065_1_color_chart_values
    elif color_space == "itu2020":
        image = load_image_itu2020(image_path)
        color_chart_values = itu_r_bt_2020_color_chart_values
    else:
        raise ValueError(f"Unknown color space: {color_space}")

    passed, failed, accuracy, _ = validate_color_chart(image, color_chart_values)
    if passed == len(color_chart_values):
        logger.info("Image passed the color check.")
    else:
        logger.error("Image failed the color check.")


def main():
    # Define a list of image paths and corresponding color spaces
    image_color_space_pairs = [
        ('image_charts/ACES2065_1_1920_1080.exr', 'aces2065_1'),
        ('image_charts/ITU-R_BT.2020.exr', 'itu2020'),
        ('image_charts/RGB_1920_1080.tif', 'rgb')
    ]

    for image_path, color_space in image_color_space_pairs:
        try:
            # Run the color chart validation for the current image and color space
            validate_single_image(image_path, color_space)
        except Exception as e:
            logger.error(f"Failed to validate color chart for image {image_path} in color space {color_space}: {e}")
            continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate color accuracy of an image.")
    parser.add_argument('--image_path', type=str, help='Path to the image file.', default=None)
    parser.add_argument('--color_space', type=str, help='Color space of the image. It should be one of "rgb", "aces2065_1", or "itu2020".', default=None)
    args = parser.parse_args()

    if args.image_path is None or args.color_space is None:
        # If no arguments are provided, run checks for all color spaces
        main()
    else:
        # If arguments are provided, run the check for the specified color space
        validate_single_image(args.image_path, args.color_space)
