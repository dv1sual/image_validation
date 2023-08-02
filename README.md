# Color Accuracy Validation

This Python script validates the color accuracy of images in different color spaces (RGB, ACES2065-1, ITU2020, ITU709, Adobe RGB) based on specified color charts. The color chart values are loaded from a JSON file, and each image's color accuracy is evaluated based on these charts.

## Dependencies

The script depends on several Python libraries:

- cv2
- numpy
- PIL
- OpenEXR
- Imath
- os
- json
- argparse

## Usage

The script can be run from the command line as follows:

python color_accuracy_validation.py --image_path /path/to/your/image.file --color_space rgb

This will run the color accuracy validation on the specified image file in the RGB color space.

You can specify a different color space by changing the 'color_space' argument to the desired one.

If no arguments are provided, the script will run the color accuracy validation for all color spaces on a set of default image paths.

## Color Accuracy Validation

The color accuracy validation is based on a color chart for each color space. Each color chart is specified in the 'color_charts_values/color_chart_values.json' file.

The script evaluates the color accuracy of an image by comparing the actual RGB values at each color patch location with the expected RGB values from the color chart. The accuracy is calculated as 1 minus the Euclidean distance between the expected and actual RGB values, divided by the maximum possible distance.

## Logging

The script logs messages about the loading and validation process to the console and to separate log files for each color space. The log files are located in the 'logs' directory and are named according to the color space and the timestamp of the run (e.g., 'rgb.log').

