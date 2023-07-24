import cv2  # For image processing
import numpy as np  # For array operations
from PIL import Image  # For image processing
import OpenEXR  # For opening EXR files
import Imath  # For handling EXR files
import os  # For handling file paths
import logging
from utils.logger_config import configure_logger  # For logging configuration
import json  # For handling JSON files


logger = logging.getLogger()


# Load color chart values from a JSON file
with open('color_charts_values/color_chart_values.json', 'r') as file:
    color_charts = json.load(file)

aces2065_1_color_chart_values = color_charts['aces2065_1_color_chart_values']
itu_r_bt_2020_color_chart_values = color_charts['itu_r_bt_2020_color_chart_values']
itu_r_bt_709_color_chart_values = color_charts['itu_r_bt_709_color_chart_values']
rgb_color_chart_values = color_charts['RGB_color_chart_values']


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


def load_image_itu709(image_path):
    """Load an ITU-R BT.709 image file and return it as a numpy array."""
    global logger
    logger = configure_logger('itu709')
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")

    filename = os.path.basename(image_path)
    logger.info("ITU-R BT.709 Color Check")
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
        logger.error(f"Loading non-EXR file for ITU-R BT.709: {image_path}")
        raise ValueError(f"Non-EXR file provided for ITU-R BT.709 color chart: {image_path}")

    logger.info("Color checking starting...")
    return image


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
        # Normalize the image array values to the range 0.0 - 1.0 if they aren't already
        if np.max(image) > 1.0:
            image = image / 255.0
    except Exception as image_load_exception:
        logger.error(f"Failed to load image: {image_load_exception}")
        raise

    logger.info(f"Successfully loaded {filename}")
    logger.info(f"Color checking starting...")
    return image


def visualize_color_chart(image, chart_values, color_space, image_path):
    """Visualize a color chart as an image."""
    canvas_height, canvas_width = 140 * len(chart_values) + 20, 500
    canvas_vis = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    for i, value in enumerate(chart_values):
        expected_rgb, actual_rgb = value["rgb"], image[value["y"], value["x"]]
        top, bottom = 10 + i * 140, 10 + i * 140 + 120
        expected_rgb_int, actual_rgb_int = [tuple([int(c * 255) for c in rgb][::-1]) for rgb in
                                            [expected_rgb, actual_rgb]]

        accuracy_percentage = value["accuracy"] * 100
        border_color = (0, 255, 0) if accuracy_percentage >= 99.97 else (0, 0, 255)

        cv2.rectangle(canvas_vis, (50, top), (150, bottom), border_color, 2)
        cv2.rectangle(canvas_vis, (250, top), (350, bottom), border_color, 2)
        cv2.rectangle(canvas_vis, (51, top + 1), (149, bottom - 1), expected_rgb_int, -1)
        cv2.rectangle(canvas_vis, (251, top + 1), (349, bottom - 1), actual_rgb_int, -1)

        cv2.putText(canvas_vis, value["name"], (10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        for j, color_name in enumerate(['R', 'G', 'B']):
            expected_text = f"{color_name}: {expected_rgb[j]:.3f}"
            actual_text = f"{color_name}: {actual_rgb[j]:.3f}"
            y_offset = top + 20 + j * 20  # Increment y position for each color component
            cv2.putText(canvas_vis, expected_text, (155, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas_vis, actual_text, (355, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{color_space}_result.png"
    output_path = os.path.join('results', output_filename)
    cv2.imwrite(output_path, canvas_vis)


def validate_color_chart(image, chart_values, color_space, image_path, tolerance=0.05):
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

    # Create a visualisation for the current color space
    output_path = f'result_{color_space}.png'  # filename based on color space
    visualize_color_chart(image, chart_values, color_space, image_path)

    return passed, failed, overall_accuracy, chart_values


def configure_loggers(color_space):
    # Initialize logger for the specified color space
    loggers = {
        'rgb': configure_logger('rgb'),
        'aces2065_1': configure_logger('aces2065_1'),
        'itu2020': configure_logger('itu2020'),
        'itu709': configure_logger('itu709')
    }
    return loggers[color_space]


def visualize_validation(image, chart_values):
    output_path = "results/"
    visualize_color_chart(image, chart_values, output_path)


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
    elif color_space == "itu709":
        image = load_image_itu709(image_path)
        color_chart_values = itu_r_bt_709_color_chart_values
    else:
        raise ValueError(f"Unknown color space: {color_space}")

    passed, failed, accuracy, chart_values = validate_color_chart(image, color_chart_values, color_space, image_path)
    if passed == len(color_chart_values):
        logger.info("Image passed the color check.")
    else:
        logger.error("Image failed the color check.")


def main():
    # Define a list of image paths and corresponding color spaces
    image_color_space_pairs = [
        ('image_charts/ACES2065_1_1920_1080.exr', 'aces2065_1'),
        ('image_charts/ITU-R_BT.2020.exr', 'itu2020'),
        ('image_charts/ITU-R_BT.709.exr', 'itu709'),
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
