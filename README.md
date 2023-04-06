# image_validation

# Color Chart Validation

This script validates an image's colors against a predefined color chart. It reports the accuracy of each color patch in the image and whether the validation passed or failed.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pillow
- colorama
- termcolor

## Usage

1. Set the correct file path for the image to be validated in the `load_image()` function.
2. Run the script with `python color_chart_validation.py`.

## Output

The script will print information about the loaded image and the validation status of each color patch. If all color patches pass the validation, a success message is displayed. Otherwise, a failure message is displayed.
