import cv2
import numpy as np
from termcolor import colored
from colorama import init, Fore
from PIL import Image
from color_charts_values.RGB_color_chart_values import color_chart_values
import os
import time
from logger_config import configure_logger

# Create a logger for the module
logger = configure_logger(__name__)


def load_image(image_path):
    """
    Load an image using PIL and print info messages.
    """
    image = Image.open(image_path)
    filename = os.path.basename(image_path)
    logger.info("Loading image, please wait...")
    logger.info(f"Successfully loaded {filename}")
    logger.info(f"Color checking starting...")
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

        if accuracy < 99.50:
            validation_result = False

        results.append((color_patch, accuracy))

    for color_patch, accuracy in results:
        if accuracy < 99.50:
            logger.error("Color patch {} failed validation with accuracy {:.2f}%".format(color_patch, accuracy))
        else:
            logger.info("Color patch {} passed validation with accuracy {:.2f}%".format(color_patch, accuracy))

    passed_checks = [result for result in results if result[1] >= 99.50]
    failed_checks = [result for result in results if result[1] < 99.50]
    overall_accuracy = sum(result[1] for result in results) / len(results)

    logger.info("Validation Summary:")
    logger.info(f"Passed checks: {len(passed_checks)}")
    logger.info(f"Failed checks: {len(failed_checks)}")
    logger.info(f"Overall accuracy: {overall_accuracy:.2f}%")

    return validation_result


def calculate_accuracy(expected_rgb, actual_rgb):
    """
    Calculates the accuracy between two RGB values.
    """
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([255, 255, 255])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def put_vertical_text(img, text, org, font, font_scale, color, thickness):
    y = org[1]
    for value in text:
        cv2.putText(img, str(value), (org[0], y), font, font_scale, color, thickness)
        y += int(cv2.getTextSize(str(value), font, font_scale, thickness)[0][1] * 1.5)


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

        # Add the expected and actual RGB values as vertical text
        text_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        put_vertical_text(canvas, expected_rgb, (155, top + 20), font, font_scale, text_color, font_thickness)
        put_vertical_text(canvas, actual_rgb, (355, top + 20), font, font_scale, text_color, font_thickness)

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
        logger.info('Color chart validation passed successfully')
    else:
        logger.info('Color chart validation failed')


if __name__ == '__main__':
    main()
