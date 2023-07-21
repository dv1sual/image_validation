import cv2  # For image processing
import numpy as np  # For array operations
import OpenEXR  # For opening EXR files
import Imath  # For handling EXR files
from color_charts_values.itu_r_bt_2020_color_chart_values import itu_r_bt_2020_color_chart_values
import os  # For handling file paths
from logger_config import configure_logger


# Create a logger for the module
logger = configure_logger(__name__)


def load_image(image_path):
    """
    This function opens an image file and returns it as a numpy array.

    :param image_path: A string containing the file path to the image.
    :return: A numpy array of the image data.
    """
    filename = os.path.basename(image_path)
    logger.info("ITU-BT2020 Color Check")
    logger.info("Loading image, please wait...")  # Log info message

    # Check if file is an EXR file
    if image_path.lower().endswith(".exr"):
        # Code for loading EXR files
        logger.debug(f"Loading EXR file: {image_path}")
        exr_file = OpenEXR.InputFile(image_path)  # This line uses the OpenEXR library to open the EXR file.
        dw = exr_file.header()['dataWindow']  # This retrieves the 'dataWindow' property from the header of the EXR file. This property contains information about the image's dimensions.
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)  # This line calculates the width and height of the image based on the 'dataWindow' property.

        pt = Imath.PixelType(Imath.PixelType.FLOAT)  # This sets up a pixel type object representing a floating point pixel. This is used to tell OpenEXR how to interpret the pixel data.
        r_str = exr_file.channel('R', pt)  # These lines extract the red, green, and blue color channels from the EXR file
        g_str = exr_file.channel('G', pt)
        b_str = exr_file.channel('B', pt)

        r_channel = np.frombuffer(r_str, dtype=np.float32).reshape((size[1], size[0]))  # These lines take the raw data from the color channels, interpret it as 32-bit floating point numbers, and reshape it into 2D arrays that match the dimensions of the image.
        g_channel = np.frombuffer(g_str, dtype=np.float32).reshape((size[1], size[0]))
        b_channel = np.frombuffer(b_str, dtype=np.float32).reshape((size[1], size[0]))

        image = np.dstack((r_channel, g_channel, b_channel))  # This stacks the 2D arrays for the red, green, and blue color channels together along a third dimension to create a 3D array representing the image.
        logger.info(f"Successfully loaded {filename}")
    else:
        # Code for loading non-EXR files
        logger.debug(f"Loading non-EXR file: {image_path}")
        image = Image.open(image_path)
        print(type(image))
        image = np.array(image) / 255.0  # Converts the image data to a numpy array and scales the pixel values from 0-255 to 0-1 by dividing by 255.

    logger.info(f"Color checking starting...")

    return image


def validate_color_chart(image, chart_values, tolerance=0.05):
    """
    This function checks the color accuracy of an image based on a color chart.

    :param image: A numpy array of the image data.
    :param chart_values: A list of dictionaries, each containing an RGB value and coordinates.
    :param tolerance: The allowed difference between the actual and expected color.
    :return: A tuple containing the number of passed and failed checks, the overall accuracy, and the chart values.
    """
    passed = 0
    failed = 0
    total_accuracy = 0

    for value in chart_values:
        expected_rgb = value["rgb"]
        x, y = value["x"], value["y"]

        # Debugging step 2: Print the coordinates used to fetch the pixel
        logger.info(f"Image dimensions: {image.shape}")
        logger.info(f"Coordinates: {x}, {y}")

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
        logger.info(f"Color name: {value['name']}")
        logger.info(f"Expected RGB: {expected_rgb}")
        logger.info(f"Actual RGB: {actual_rgb}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        # print("\n")

    # Calculate the overall accuracy
    overall_accuracy = total_accuracy / len(chart_values)

    return passed, failed, overall_accuracy, chart_values


def calculate_accuracy(expected_rgb, actual_rgb):
    """
    This function calculates the accuracy between two RGB values.

    :param expected_rgb: A tuple or list of the expected RGB values.
    :param actual_rgb: A tuple or list of the actual RGB values.
    :return: A float representing the accuracy of the actual color.
    """
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([1.0, 1.0, 1.0])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def put_vertical_text(image, text, position, font, font_scale, color, thickness):
    """
    This function puts vertical text into the resulting image.

    :param image: The image to put the text into.
    :param text: The text to put into the image.
    :param position: The position where the text should start.
    :param font: The font of the text.
    :param font_scale: The scale of the font.
    :param color: The color of the font.
    :param thickness: The thickness of the font.
    """
    lines = text.split('\n')
    line_height = cv2.getTextSize(text[0], font, font_scale, thickness)[0][1]
    for i, line in enumerate(lines):
        y = position[1] + i * line_height
        cv2.putText(image, line, (position[0], y), font, font_scale, color, thickness, cv2.LINE_AA)


def visualize_color_chart(image, chart_values, output_path):
    """
    This function visualizes a color chart as an image.

    :param image: A numpy array of the image data.
    :param chart_values: A list of dictionaries, each containing an RGB value and coordinates.
    :param output_path: The file path where the output image should be saved.
    """
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
    """
    The main function of the script. It orchestrates the loading, validation, and visualization of the color chart.
    """

    # Load the image from a file
    input_image_path = 'image_charts/ITU-R_BT.2020.exr'
    image = load_image(input_image_path)

    if "ITU-R_BT.2020" in input_image_path:
        chart_values = itu_r_bt_2020_color_chart_values
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
        logger.info('Color chart validation passed successfully')
    else:
        logger.error('Color chart validation failed')


if __name__ == '__main__':
    main()
