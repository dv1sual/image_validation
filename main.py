import cv2
import numpy as np
from termcolor import colored
from colorama import init, Fore
from PIL import Image
from color_chart_values import color_chart_values
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
    Load an image using PIL and print info messages.
    """
    image = Image.open(image_path)
    filename = os.path.basename(image_path)
    print(colored_info("Loading image, please wait..."))
    time.sleep(2)
    print(colored_info(f"Successfully loaded {filename}"))
    time.sleep(2)
    print(colored_info(f"Color checking starting..."))
    time.sleep(2)
    return image


def validate_color_chart(image, chart_values):
    """
    Validates the colors in the image against the provided color chart values.
    """
    validation_result = True
    opencv_image = np.array(image)

    results = []
    for color_patch in chart_values:
        x, y, expected_rgb = color_patch['x'], color_patch['y'], color_patch['rgb']
        actual_rgb = opencv_image[y, x]
        accuracy = calculate_accuracy(expected_rgb, actual_rgb)

        if accuracy < 99.99:
            validation_result = False

        results.append((color_patch, accuracy))

    for color_patch, accuracy in results:
        if accuracy < 99.99:
            print(colored("[CHECK] - Color patch {} failed validation with accuracy {:.2f}%".format(color_patch, accuracy), "red"))
        else:
            print(colored("[CHECK] - Color patch {} passed validation with accuracy {:.2f}%".format(color_patch, accuracy), "green"))

        time.sleep(2)

    passed_checks = [result for result in results if result[1] >= 99.99]
    failed_checks = [result for result in results if result[1] < 99.99]
    overall_accuracy = sum(result[1] for result in results) / len(results)

    print("\nValidation Summary:")
    print(f"Passed checks: {len(passed_checks)}")
    print(f"Failed checks: {len(failed_checks)}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%\n")

    return validation_result



def calculate_accuracy(expected_rgb, actual_rgb):
    """
    Calculates the accuracy between two RGB values.
    """
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([255, 255, 255])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def visualize_color_chart(image, chart_values, output_filename):
    # Create a blank canvas with a white background
    canvas_height = (len(chart_values) * 100) + (len(chart_values) * 20)
    canvas = np.ones((canvas_height, 400, 3), dtype=np.uint8) * 255

    # Iterate through the color patches
    for idx, color_patch in enumerate(chart_values):
        x, y, expected_rgb = color_patch['x'], color_patch['y'], color_patch['rgb']
        actual_rgb = tuple(map(int, image[y, x]))
        accuracy = calculate_accuracy(expected_rgb, actual_rgb)

        # Calculate the position of the color patches on the canvas
        top = idx * 120
        bottom = top + 100

        # Draw the expected and actual color patches side by side
        cv2.rectangle(canvas, (50, top), (150, bottom), tuple(expected_rgb[::-1]), -1)
        cv2.rectangle(canvas, (250, top), (350, bottom), tuple(actual_rgb[::-1]), -1)

        # Highlight the patches with a border (green for pass, red for fail)
        border_color = (0, 255, 0) if accuracy >= 99.99 else (0, 0, 255)
        cv2.rectangle(canvas, (50, top), (150, bottom), border_color, 2)
        cv2.rectangle(canvas, (250, top), (350, bottom), border_color, 2)

    # Save the visualization to a file
    cv2.imwrite(output_filename, canvas)


def main():
    # Load the image from a file
    input_image_path = 'image_charts/RGB_1920_1080.tif'
    image = load_image(input_image_path)
    validation_result = validate_color_chart(image, color_chart_values)

    # Create the output filename and save the visualization to a specific folder
    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + '_results.png'
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    visualize_color_chart(np.array(image), color_chart_values, output_path)

    if validation_result:
        print(colored_info('Color chart validation passed successfully'))
    else:
        print(colored_info('Color chart validation failed'))


if __name__ == '__main__':
    main()
