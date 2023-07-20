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
        r_str = exr_file.channel('R', pt)
        g_str = exr_file.channel('G', pt)
        b_str = exr_file.channel('B', pt)

        r_channel = np.frombuffer(r_str, dtype=np.float32).reshape((size[1], size[0]))
        g_channel = np.frombuffer(g_str, dtype=np.float32).reshape((size[1], size[0]))
        b_channel = np.frombuffer(b_str, dtype=np.float32).reshape((size[1], size[0]))

        image = np.dstack((r_channel, g_channel, b_channel))
        print(colored_info(f"Successfully loaded {filename}"))
    else:
        image = Image.open(image_path)
        print(type(image))
        image = np.array(image) / 255.0  # Convert the non-EXR images to floating-point values in the 0-1 range

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

        # Debugging step 2: Print the coordinates used to fetch the pixel
        print(f"Image dimensions: {image.shape}")
        print(f"Coordinates: {x}, {y}")

        actual_rgb = image[y, x]

        # Calculate the accuracy
        accuracy = calculate_accuracy(expected_rgb, actual_rgb) / 100  # calculate_accuracy returns a percentage, so divide by 100

        # Check if the accuracy is within the tolerance
        if accuracy >= (1 - tolerance):
            passed += 1
            total_accuracy += accuracy
        else:
            failed += 1

        # Update the chart_values dictionary with accuracy
        value["accuracy"] = accuracy

        # Debug information
        print(f"Color name: {value['name']}")
        print(f"Expected RGB: {expected_rgb}")
        print(f"Actual RGB: {actual_rgb}")
        print(f"Accuracy: {accuracy:.2%}")
        print("\n")

    # Calculate the overall accuracy
    overall_accuracy = total_accuracy / len(chart_values)

    return passed, failed, overall_accuracy, chart_values


def calculate_accuracy(expected_rgb, actual_rgb):
    """
    Calculates the accuracy between two RGB values.
    """
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([1.0, 1.0, 1.0])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def put_vertical_text(image, text, position, font, font_scale, color, thickness):
    lines = text.split('\n')
    line_height = cv2.getTextSize(text[0], font, font_scale, thickness)[0][1]
    for i, line in enumerate(lines):
        y = position[1] + i * line_height
        cv2.putText(image, line, (position[0], y), font, font_scale, color, thickness, cv2.LINE_AA)


def visualize_color_chart(image, chart_values, output_path):
    # Create an 8-bit integer canvas with a white background for visualization
    canvas_height = 140 * len(chart_values) + 20  # Increased height to accommodate the text
    canvas_width = 500  # Increased width to accommodate the text
    canvas_vis = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Draw the color patches and annotations
    for i, value in enumerate(chart_values):
        expected_rgb = value["rgb"]
        x, y = value["x"], value["y"]
        actual_rgb = image[y, x]

        # Round the actual RGB values to the 3rd decimal place
        actual_rgb_rounded = tuple([round(c, 3) for c in actual_rgb])

        # Calculate the position of the color patches
        top = 10 + i * 140
        bottom = top + 120

        # Convert the floating-point RGB values to 0-255 integers for drawing
        expected_rgb_int = tuple([int(c * 255) for c in expected_rgb][::-1])  # Reverse the order for BGR
        actual_rgb_int = tuple([int(c * 255) for c in actual_rgb][::-1])  # Reverse the order for BGR

        # Get the accuracy from the chart_values dictionary
        accuracy_percentage = value["accuracy"] * 100  # Convert accuracy back to a percentage for display

        # Determine the border color based on accuracy
        threshold = 99.97
        if accuracy_percentage >= threshold:
            border_color = (0, 255, 0)  # Green for pass
        else:
            border_color = (0, 0, 255)  # Red for fail

        # Draw the border around the color patches
        cv2.rectangle(canvas_vis, (50, top), (150, bottom), border_color, 2)
        cv2.rectangle(canvas_vis, (250, top), (350, bottom), border_color, 2)

        # Fill the color patches with the expected and actual RGB colors
        cv2.rectangle(canvas_vis, (51, top + 1), (149, bottom - 1), expected_rgb_int, -1)
        cv2.rectangle(canvas_vis, (251, top + 1), (349, bottom - 1), actual_rgb_int, -1)

        # Add the color patch name
        name_position = (10, top + 20)
        cv2.putText(canvas_vis, value["name"], name_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Add the expected and actual RGB values as vertical text
        text_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        expected_text = f"R: {expected_rgb[0]:.3f}\nG: {expected_rgb[1]:.3f}\nB: {expected_rgb[2]:.3f}"
        actual_text = f"R: {actual_rgb_rounded[0]:.3f}\nG: {actual_rgb_rounded[1]:.3f}\nB: {actual_rgb_rounded[2]:.3f}"
        put_vertical_text(canvas_vis, expected_text, (155, top + 20), font, font_scale, text_color, font_thickness)
        put_vertical_text(canvas_vis, actual_text, (355, top + 20), font, font_scale, text_color, font_thickness)

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
    passed, failed, overall_accuracy, chart_values_with_accuracy = validate_color_chart(image, chart_values)

    # Create the output filename and save the visualization to a specific folder
    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + '_results.png'
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    visualize_color_chart(np.array(image), chart_values_with_accuracy, output_path)

    if validation_result[2] > 0.9:  # Let's assume validation passed if overall accuracy is more than 90%
        print(colored_info('Color chart validation passed successfully'))
    else:
        print(colored_info('Color chart validation failed'))


if __name__ == '__main__':
    main()
