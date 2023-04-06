import cv2
import numpy as np
import time
from termcolor import colored
from colorama import init, Fore
from PIL import Image
from color_chart_values import color_chart_values

# Initialize colorama
init(autoreset=True)

# Set cyan color for CHECK messages
CHECK_COLOR = Fore.CYAN


def colored_info(text):
    return colored(f"[INFO] - {text}", "red")


def load_image(image_path):
    image = Image.open(image_path)
    return image


def validate_color_chart(image, color_chart_values):
    validation_result = True
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for color_patch in color_chart_values:
        x, y, expected_rgb = color_patch['x'], color_patch['y'], color_patch['rgb']
        print(CHECK_COLOR + "[CHECK] - Checking color patch {} at coordinates ({}, {})".format(color_patch, x, y))
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
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([255, 255, 255])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def main():
    # Load the image from a file
    opencv_image = cv2.imread("Untitled-1.png", cv2.IMREAD_COLOR)
    color_chart_coords = None  # Set to None, as it's not needed anymore

    # Capture screenshot from selected output
    print(colored_info("Loading image"))
    image = load_image('Macbeth_ColorChart.tif')  # Replace with the actual file path
    time.sleep(2)

    validation_result = validate_color_chart(opencv_image, color_chart_values)

    if validation_result:
        print(colored_info('Color chart found and validated'))
    else:
        print(colored_info('Color chart found but validation failed'))


if __name__ == '__main__':
    main()

