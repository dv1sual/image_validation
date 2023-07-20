import cv2
import numpy as np
from termcolor import colored
from colorama import init, Fore
from PIL import Image
import OpenEXR
import Imath
from color_charts_values.RGB_color_chart_values import color_chart_values
from color_charts_values.aces2065_1_color_chart_values import aces2065_1_color_chart_values
import os
import time

# Initialize colorama
init(autoreset=True)

# Set cyan color for CHECK messages
CHECK_COLOR = Fore.CYAN


def colored_info(text):
    """
    Returns a red-colored text for displaying info messages.
    """
    return colored(f"[INFO] - {text}", "red")


def load_image(image_path):
    """
    Load an image using OpenEXR and Imath for EXR images, and PIL for other formats.
    """
    filename = os.path.basename(image_path)
    print(colored_info("Loading image, please wait..."))
    time.sleep(2)

    if image_path.lower().endswith(".exr"):
        exr_file = OpenEXR.InputFile(image_path)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r_channel, g_channel, b_channel = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in ("R", "G", "B")]
        r_channel.shape = g_channel.shape = b_channel.shape = size

        image = np.stack((r_channel, g_channel, b_channel), axis=-1)
        image = np.clip(image, 0, 1)  # Keep the values in the 0-1 floating-point range
    else:
        image = Image.open(image_path)
        print(type(image))
        image = np.array(image) / 255.0  # Convert the non-EXR images to floating-point values in the 0-1 range

    print(colored_info(f"Successfully loaded {filename}"))
    time.sleep(2)
    print(colored_info(f"Color checking starting..."))
    time.sleep(2)
    return image


def validate_color_chart(image, chart_values, tolerance=0.05):
    passed = 0
    failed = 0
    total_accuracy = 0

    for value in chart_values:
        expected_rgb = value["rgb"]
        x, y = value["x"], value["y"]

        actual_rgb = image[x, y]

        # Calculate the color difference
        color_diff = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
        max_diff = np.max(color_diff)

        # Check if the color difference is within the tolerance
        if max_diff <= tolerance:
            passed += 1
            accuracy = 1 - max_diff / tolerance
            total_accuracy += accuracy
        else:
            failed += 1
            accuracy = 0

        # Debug information
        print(f"Color name: {value['name']}")
        print(f"Expected RGB: {expected_rgb}")
        print(f"Actual RGB: {actual_rgb}")
        print(f"Color difference: {color_diff}")
        print(f"Max difference: {max_diff}")
        print(f"Accuracy: {accuracy:.2%}")
        print("\n")

    # Calculate the overall accuracy
    overall_accuracy = total_accuracy / len(chart_values)

    return passed, failed, overall_accuracy


def calculate_accuracy(expected_rgb, actual_rgb):
    """
    Calculates the accuracy between two RGB values.
    """
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([1.0, 1.0, 1.0])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def visualize_color_chart(image, chart_values, output_path):
    # Create an 8-bit integer canvas with a white background for visualization
    canvas_height = 120 * len(chart_values) + 20
    canvas_width = 400
    canvas_vis = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Draw the color patches
    for i, value in enumerate(chart_values):
        expected_rgb = value["rgb"]
        x, y = value["x"], value["y"]
        actual_rgb = image[x, y]

        # Calculate the position of the color patches
        top = 10 + i * 120
        bottom = top + 100

        # Convert the floating-point RGB values to 0-255 integers for drawing
        expected_rgb_int = tuple([int(c * 255) for c in expected_rgb])
        actual_rgb_int = tuple([int(c * 255) for c in actual_rgb])

        # Draw the expected and actual color patches side by side
        cv2.rectangle(canvas_vis, (50, top), (150, bottom), expected_rgb_int[::-1], -1)
        cv2.rectangle(canvas_vis, (250, top), (350, bottom), actual_rgb_int[::-1], -1)

        # Add the color patch name
        text_position = (10, (top + bottom) // 2)
        cv2.putText(canvas_vis, value["name"], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the visualization canvas as a PNG file
    cv2.imwrite(output_path, canvas_vis)


def main():
    # Load the image from a file
    input_image_path = 'image_charts/ACES2065_1_1920_1080.exr'
    image = load_image(input_image_path)

    if "ACES2065_1" in input_image_path:
        chart_values = aces2065_1_color_chart_values
    else:
        chart_values = color_chart_values

    validation_result = validate_color_chart(image, chart_values)

    # Create the output filename and save the visualization to a specific folder
    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + '_results.png'
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    visualize_color_chart(np.array(image), chart_values, output_path)

    if validation_result[2] > 0.9:  # Let's assume validation passed if overall accuracy is more than 90%
        print(colored_info('Color chart validation passed successfully'))
    else:
        print(colored_info('Color chart validation failed'))


if __name__ == '__main__':
    main()
