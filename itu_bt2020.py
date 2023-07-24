import cv2  # For image processing
import numpy as np  # For array operations
import OpenEXR  # For opening EXR files
import Imath  # For handling EXR files
import os  # For handling file paths
from logger_config import configure_logger  # For logging configuration
import json  # For handling JSON files

logger = configure_logger('itu2020', 'itu2020.log')

with open('color_charts_values/color_chart_values.json', 'r') as file:
    color_charts = json.load(file)

itu_r_bt_2020_color_chart_values = color_charts['itu_r_bt_2020_color_chart_values']


def load_image(image_path):
    """Load an image file and return it as a numpy array."""
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

    for value in chart_values:
        if not all(key in value for key in ["rgb", "x", "y"]):
            raise ValueError("Each color patch dictionary must have 'rgb', 'x', and 'y' keys.")

    passed, failed, total_accuracy = 0, 0, 0

    for value in chart_values:
        expected_rgb = value["rgb"]
        x, y = value["x"], value["y"]

        actual_rgb = image[y, x]
        accuracy = calculate_accuracy(expected_rgb, actual_rgb) / 100

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
    return passed, failed, overall_accuracy, chart_values


def calculate_accuracy(expected_rgb, actual_rgb):
    """Calculate the accuracy between two RGB values."""
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([1.0, 1.0, 1.0])
    return 100 - (np.mean(differences / max_differences) * 100)


def visualize_color_chart(image, chart_values, output_path):
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

    cv2.imwrite(output_path, canvas_vis)


def main():
    input_image_path = 'image_charts/ITU-R_BT.2020.exr'
    image = load_image(input_image_path)

    if "ITU-R_BT.2020" in input_image_path:
        chart_values = itu_r_bt_2020_color_chart_values
    else:
        raise ValueError("The image file does not match with any known color chart.")

    passed, failed, overall_accuracy, chart_values_with_accuracy = validate_color_chart(image, chart_values)

    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + '_results.png'
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    visualize_color_chart(np.array(image), chart_values_with_accuracy, output_path)

    if overall_accuracy > 0.9:  # The condition here was changed from validation_result[2] > 0.9
        logger.info('Color chart validation passed successfully')
    else:
        logger.error('Color chart validation failed')


if __name__ == '__main__':
    main()
