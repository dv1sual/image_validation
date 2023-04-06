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

    for color_patch in chart_values:
        x, y, expected_rgb = color_patch['x'], color_patch['y'], color_patch['rgb']
        actual_rgb = opencv_image[y, x]
        accuracy = calculate_accuracy(expected_rgb, actual_rgb)

        if accuracy < 99.99:
            print(colored("[CHECK] - Color patch {} failed validation with accuracy {:.2f}%".format(color_patch, accuracy), "red"))
            validation_result = False
        else:
            print(colored("[CHECK] - Color patch {} passed validation with accuracy {:.2f}%".format(color_patch, accuracy), "green"))

        time.sleep(3)

    return validation_result


def calculate_accuracy(expected_rgb, actual_rgb):
    """
    Calculates the accuracy between two RGB values.
    """
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([255, 255, 255])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def main():
    # Load the image from a file
    image = load_image('image_charts/chart_1920_1080.tif')  # Replace with the actual file path
    validation_result = validate_color_chart(image, color_chart_values)

    if validation_result:
        print(colored_info('Color chart validation passed successfully'))
    else:
        print(colored_info('Color chart validation failed'))


if __name__ == '__main__':
    main()

