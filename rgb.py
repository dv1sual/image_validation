import cv2  # For image processing
import numpy as np  # For array operations
from PIL import Image  # For image processing
import os  # For handling file paths
from utils.logger_config import configure_logger  # For logging configuration
import json  # For handling JSON files

# Create a logger for the module
logger = configure_logger('rgb', 'rgb.log')

# Load color chart values from a JSON file
try:
    with open('color_charts_values/color_chart_values.json', 'r') as file:
        color_charts = json.load(file)
except Exception as json_load_exception:
    logger.error(f"Failed to load color chart values: {json_load_exception}")
    raise

color_chart_values = color_charts['RGB_color_chart_values']


def load_image(image_path):
    """
    Load an image using PIL and print info messages.
    """
    try:
        image = Image.open(image_path)
    except Exception as image_load_exception:
        logger.error(f"Failed to load image: {image_load_exception}")
        raise

    filename = os.path.basename(image_path)
    logger.info("RGB Color Check")
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

        logger.info("Color name: {}".format(color_patch['name']))
        logger.info("Expected RGB: [{:.3f}, {:.3f}, {:.3f}]".format(*[value/255 for value in expected_rgb]))
        logger.info("Actual RGB: {}".format(actual_rgb/255))
        logger.info("Accuracy: {:.2f}%".format(accuracy))

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

        y = top + 20
        for value in expected_rgb:
            cv2.putText(canvas, str(value), (155, y), font, font_scale, text_color, font_thickness)
            y += int(cv2.getTextSize(str(value), font, font_scale, font_thickness)[0][1] * 1.5)

        y = top + 20
        for value in actual_rgb:
            cv2.putText(canvas, str(value), (355, y), font, font_scale, text_color, font_thickness)
            y += int(cv2.getTextSize(str(value), font, font_scale, font_thickness)[0][1] * 1.5)

    # Save the visualization to a file
    cv2.imwrite(output_filename, canvas)


def main():
    # Load the image from a file
    input_image_path = 'image_charts/RGB_1920_1080.tif'
    image = load_image(input_image_path)
    logger.info("RGB Color Check")
    logger.info("Loading image...")
    logger.info(f"Successfully loaded {os.path.basename(input_image_path)}")
    logger.info("Color checking starting...")

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
