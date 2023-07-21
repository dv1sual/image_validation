# Image Color Accuracy Validation Script

## Overview

This script is designed to validate the color accuracy of images based on a provided color chart. It accepts images in various formats, including .exr, and returns a visual representation of the results, which showcases the expected vs actual colors and their accuracy. The accuracy is calculated based on the difference between the actual and expected RGB values.

## Prerequisites

Python 3.6 or later

OpenCV for Python (pip install opencv-python)

numpy (pip install numpy)

Pillow (pip install Pillow)

OpenEXR (pip install OpenEXR)

Imath (pip install imath)

## Python packages:

cv2: For image processing.

numpy: For array operations.

PIL (Pillow): For opening image files.

OpenEXR: For opening EXR files.

Imath: For handling EXR files.

It also requires a color chart values, which can be found in the color_charts_values directory.

## Usage
To use the image color accuracy validation script, follow these steps:

Ensure that the prerequisites are installed and the required color chart values are in the correct directory.
In your Python environment, run the script with python <script_name>.py.
The script will load the image, validate it against the color chart, and save a visualization of the results in the results directory. The visualization shows the expected and actual colors and their accuracy.
