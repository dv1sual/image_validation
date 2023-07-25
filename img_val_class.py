import cv2  # For image processing
import numpy as np  # For array operations
from PIL import Image  # For image processing
import OpenEXR  # For opening EXR files
import Imath  # For handling EXR files
import os  # For handling file paths
import logging  # For logging
from utils.logger_config import configure_logger  # For logging configuration
import json  # For handling JSON files


class ColorChecker:
    def __init__(self, image_path, color_space):
        """
        Initialize a ColorChecker object.

        :param image_path: The path to the image file.
        :type image_path: str
        :param color_space: The color space of the image.
        :type color_space: str
        """
        self.image_path = image_path
        self.color_space = color_space
        self.logger = self.configure_logger(color_space)

        try:
            with open('color_charts_values/color_chart_values.json', 'r') as file:
                self.color_charts = json.load(file)
            with open('config/color_space_config.json', 'r') as file:
                self.color_space_config = json.load(file)
        except Exception as e:
            self.logger.error(f"Failed to load color chart values or color space config: {str(e)}")
            raise

        if self.color_space not in self.color_space_config:
            raise ValueError(f"Unknown color space: {self.color_space}")

    @staticmethod
    def load_exr_image(image_path):
        """
        Load an EXR image file and return it as a numpy array.

        :param image_path: The path to the EXR image file.
        :type image_path: str
        :return: The image data as a numpy array.
        :rtype: numpy.ndarray
        """
        exr_file = OpenEXR.InputFile(image_path)
        dw = exr_file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        r_str, g_str, b_str = exr_file.channel('R', pt), exr_file.channel('G', pt), exr_file.channel('B', pt)

        r_channel, g_channel, b_channel = [np.frombuffer(channel_str, dtype=np.float32).reshape((size[1], size[0])) for
                                           channel_str in [r_str, g_str, b_str]]
        return np.dstack((r_channel, g_channel, b_channel))

    def load_image_aces(self, image_path, logger):
        """
        Load an ACES image file and return it as a numpy array. If the image file
        is not an EXR file, it raises a ValueError. It logs the progress of loading
        the image and the start of the color checking process.

        :param image_path: The path to the ACES image file.
        :type image_path: str
        :param logger: The logger to log the progress.
        :type logger: logging.Logger
        :return: The image data as a numpy array.
        :rtype: numpy.ndarray
        :raises FileNotFoundError: If the image file does not exist.
        :raises ValueError: If the image file is not an EXR file.
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The image file {image_path} does not exist.")

            filename = os.path.basename(image_path)
            logger.info("ACES2065_1 Color Check")
            logger.info("Loading image...")

            if image_path.lower().endswith(".exr"):
                image = self.load_exr_image(image_path)
                logger.info(f"Successfully loaded {filename}")
            else:
                logger.error(f"Loading non-EXR file for ACES: {image_path}")
                raise ValueError(f"Non-EXR file provided for ACES color chart: {image_path}")

            logger.info("Color checking starting...")
            return image
        except Exception as e:
            logger.error(f"Failed to load ACES image: {str(e)}")
            raise

    def load_image_itu2020(self, image_path, logger):
        """
        Load an ITU2020 image file and return it as a numpy array. If the image file
        is not an EXR file, it raises a ValueError. It logs the progress of loading
        the image and the start of the color checking process.

        :param image_path: The path to the ITU2020 image file.
        :type image_path: str
        :param logger: The logger to log the progress.
        :type logger: logging.Logger
        :return: The image data as a numpy array.
        :rtype: numpy.ndarray
        :raises FileNotFoundError: If the image file does not exist.
        :raises ValueError: If the image file is not an EXR file.
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The image file {image_path} does not exist.")

            filename = os.path.basename(image_path)
            logger.info("ITU-R_BT.2020 Color Check")
            logger.info("Loading image...")

            if image_path.lower().endswith(".exr"):
                image = self.load_exr_image(image_path)
                logger.info(f"Successfully loaded {filename}")
            else:
                logger.error(f"Loading non-EXR file for ITU2020: {image_path}")
                raise ValueError(f"Non-EXR file provided for ITU2020 color chart: {image_path}")

            logger.info("Color checking starting...")
            return image
        except Exception as e:
            logger.error(f"Failed to load ITU2020 image: {str(e)}")
            raise

    def load_image_itu709(self, image_path, logger):
        """
        Load an ITU709 image file and return it as a numpy array. If the image file
        is not an EXR file, it raises a ValueError. It logs the progress of loading
        the image and the start of the color checking process.

        :param image_path: The path to the ITU709 image file.
        :type image_path: str
        :param logger: The logger to log the progress.
        :type logger: logging.Logger
        :return: The image data as a numpy array.
        :rtype: numpy.ndarray
        :raises FileNotFoundError: If the image file does not exist.
        :raises ValueError: If the image file is not an EXR file.
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The image file {image_path} does not exist.")

            filename = os.path.basename(image_path)
            logger.info("ITU-R_BT.709 Color Check")
            logger.info("Loading image...")

            if image_path.lower().endswith(".exr"):
                image = self.load_exr_image(image_path)
                logger.info(f"Successfully loaded {filename}")
            else:
                logger.error(f"Loading non-EXR file for ITU709: {image_path}")
                raise ValueError(f"Non-EXR file provided for ITU709 color chart: {image_path}")

            logger.info("Color checking starting...")
            return image
        except Exception as e:
            logger.error(f"Failed to load ITU709 image: {str(e)}")
            raise

    @staticmethod
    def load_image_rgb(image_path, logger):
        """
        Load an RGB image file and return it as a numpy array. The image data is
        normalized to the range 0.0 - 1.0 if it isn't already. It logs the progress
        of loading the image and the start of the color checking process.

        :param image_path: The path to the RGB image file.
        :type image_path: str
        :param logger: The logger to log the progress.
        :type logger: logging.Logger
        :return: The image data as a numpy array.
        :rtype: numpy.ndarray
        :raises FileNotFoundError: If the image file does not exist.
        :raises Exception: If an error occurred during loading the image.
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The image file {image_path} does not exist.")

            filename = os.path.basename(image_path)
            logger.info("RGB Color Check")
            logger.info("Loading image, please wait...")

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

    @staticmethod
    def visualize_color_chart(image, chart_values, color_space, image_path):
        """
        Visualize a color chart in an image, and save the visualization as a new image file.
        For each color in the chart, it draws two patches of color: one for the expected color and one for the actual color.
        It also writes the name of the color and the RGB values of the expected and actual colors next to the patches.
        The visualization is saved as a PNG file in the 'results' directory.

        :param image: The image data as a numpy array.
        :type image: numpy.ndarray
        :param chart_values: A list of dictionaries, each containing information about a color in the chart.
        :type chart_values: list of dict
        :param color_space: The color space of the image. It is used in the filename of the saved visualization.
        :type color_space: str
        :param image_path: The path to the image file. It is used in the filename of the saved visualization.
        :type image_path: str
        """
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

            cv2.putText(canvas_vis, value["name"], (10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

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

    def validate_color_chart(self, image, chart_values, color_space, image_path, logger, tolerance=0.05):
        """
        Validates the color accuracy of an image based on a color chart.
        It calculates the Euclidean distance between the expected and actual RGB values for each color in the chart,
        and determines whether each color passes or fails based on a tolerance value.
        It also computes the overall accuracy of the image.

        :param image: The image data as a numpy array.
        :type image: numpy.ndarray
        :param chart_values: A list of dictionaries, each containing information about a color in the chart.
        :type chart_values: list of dict
        :param color_space: The color space of the image.
        :type color_space: str
        :param image_path: The path to the image file.
        :type image_path: str
        :param logger: The logger to use for printing information and errors.
        :type logger: logging.Logger
        :param tolerance: The tolerance value for determining whether a color passes or fails, default is 0.05.
        :type tolerance: float
        :return: A tuple containing the number of colors that passed, the number that failed, the overall accuracy, and the updated chart values.
        :rtype: tuple
        """
        try:
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
            self.visualize_color_chart(image, chart_values, color_space, image_path)

            return passed, failed, overall_accuracy, chart_values
        except Exception as e:
            logger.error(f"Failed to validate color chart: {str(e)}")
            raise

    @staticmethod
    def configure_logger(color_space):
        """
        Configures and returns a logger for the specified color space.

        :param color_space: The color space for which to configure the logger. Should be one of 'rgb', 'aces2065_1', 'itu2020', 'itu709'.
        :type color_space: str
        :return: A logger configured for the specified color space.
        :rtype: logging.Logger
        """
        loggers = {
            'rgb': configure_logger('rgb'),
            'aces2065_1': configure_logger('aces2065_1'),
            'itu2020': configure_logger('itu2020'),
            'itu709': configure_logger('itu709')
        }
        return loggers[color_space]

    def run(self):
        """
        Executes the color checking process for the image with the color space specified during the object's initialization.

        The method will:
        - Load the image.
        - Fetch the color chart values associated with the color space.
        - Validate the color chart of the image.
        - Log the result of the color check.

        If the color space is not recognized, the method will raise a ValueError.
        """
        logger = self.logger
        config = self.color_space_config[self.color_space]

        load_image_method = getattr(self, config['load_image'])
        image = load_image_method(self.image_path, logger)
        color_chart_values = self.color_charts[config['color_chart_values']]

        passed, failed, accuracy, chart_values = self.validate_color_chart(image, color_chart_values, self.color_space,
                                                                           self.image_path, logger)
        if passed == len(color_chart_values):
            logger.info("Image passed the color check.")
        else:
            logger.error("Image failed the color check.")


def main():
    """
    Main driver function to perform color checking for all color spaces (RGB, ACES, ITU2020, ITU709).

    It initializes the ColorChecker object for each color space and executes the color check by calling the run method.
    This function does not take any parameters or return any values.
    """
    rgb_checker = ColorChecker('image_charts/RGB.tif', 'rgb')
    aces_checker = ColorChecker('image_charts/ACES2065_1_1920_1080.exr', 'aces2065_1')
    itu2020_checker = ColorChecker('image_charts/ITU-R_BT.2020.exr', 'itu2020')
    itu709_checker = ColorChecker('image_charts/ITU-R_BT.709.exr', 'itu709')

    # Run the color check for each color space
    for checker in [rgb_checker, aces_checker, itu2020_checker, itu709_checker]:
        checker.run()


def validate_single_image(image_path, color_space):
    """
    Function to perform color checking for a single image in a specified color space.

    Parameters:
    image_path (str): Path to the image file.
    color_space (str): Color space of the image. It should be one of "rgb", "aces2065_1", "itu2020", or "itu709".

    Returns:
    None
    """
    checker = ColorChecker(image_path, color_space)

    # Run the color check
    checker.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate color accuracy of an image.")
    parser.add_argument('--image_path', type=str, help='Path to the image file.', default=None)
    parser.add_argument('--color_space', type=str, help='Color space of the image. It should be one of "rgb", "aces2065_1", "itu2020", or "itu709".', default=None)
    args = parser.parse_args()

    if args.image_path is None or args.color_space is None:
        # If no arguments are provided, run checks for all color spaces
        main()
    else:
        # If arguments are provided, run the check for the specified color space
        validate_single_image(args.image_path, args.color_space)
