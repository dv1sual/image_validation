import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
from termcolor import colored
from screeninfo import get_monitors
from colorama import init
from PIL import Image
import win32con
import win32gui
import win32print
from color_chart_values import color_chart_values


init(autoreset=True)


def colored_info(text):
    return colored(f"[INFO] - {text}", "red")


def capture_screenshot(monitor):
    x, y, width, height = monitor.x, monitor.y, monitor.width, monitor.height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot


def estimate_bit_depth_per_channel(image):
    channels = len(image.getbands())
    bit_depths = []

    for i in range(channels):
        channel_data = image.getchannel(i)
        unique_values = len(set(channel_data.getdata()))

        if unique_values > 256:
            bit_depths.append(10)  # Assuming 10-bit depth if unique values > 256
        else:
            bit_depths.append(8)  # Default to 8-bit depth

    return bit_depths


def get_image_info(image):
    color_space = image.mode
    channels = len(image.getbands())
    bit_depths = estimate_bit_depth_per_channel(image)
    total_bit_depth = sum(bit_depths)

    return color_space, total_bit_depth


def find_color_chart(image, color_chart_template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_color_chart = cv2.cvtColor(color_chart_template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_image, gray_color_chart, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > 0.7:
        top_left = max_loc
        bottom_right = (top_left[0] + color_chart_template.shape[1], top_left[1] + color_chart_template.shape[0])
        return top_left, bottom_right
    else:
        return None


def validate_color_chart(image, color_chart_coords):
    image = Image.fromarray(image)
    accuracy_threshold = 99.99
    top_left, _ = color_chart_coords
    failed_patches = 0

    for patch in color_chart_values:
        x = top_left[0] + patch['x']
        y = top_left[1] + patch['y']

        expected_rgb = patch['rgb']
        actual_rgb = image.getpixel((x, y))
        accuracy = calculate_accuracy(expected_rgb, actual_rgb)

        print(f"Checking color patch {patch} at coordinates ({x}, {y})")
        time.sleep(1)  # Add time.sleep() between each color patch validation

        if accuracy < accuracy_threshold:
            print(f"Color patch {patch} failed validation with accuracy {accuracy:.2f}%")
            failed_patches += 1
        else:
            print(f"Color patch {patch} passed validation with accuracy {accuracy:.2f}%")

    if failed_patches > 0:
        print(f"{failed_patches} color patches failed validation.")
        return False
    else:
        print("All color patches passed validation.")
        return True


def calculate_accuracy(expected_rgb, actual_rgb):
    differences = np.abs(np.array(expected_rgb) - np.array(actual_rgb))
    max_differences = np.array([255, 255, 255])
    accuracy = 100 - (np.mean(differences / max_differences) * 100)
    return accuracy


def list_monitors():
    monitors = get_monitors()
    for i, monitor in enumerate(monitors):
        print(colored_info(f"Output {i}: {monitor}"))


def main():
    global scaling_factor
    delay_time = 2  # Set your desired delay time in seconds

    # List available outputs
    print(colored_info("Listing outputs, please wait..."))
    time.sleep(delay_time)
    list_monitors()
    time.sleep(delay_time)

    # Prompt user to choose an output (0-based)
    monitor_index = int(input(colored_info("Enter the output number to capture (0-based): ")))
    if monitor_index < 0 or monitor_index >= len(get_monitors()):
        print(colored_info("Invalid output number"))
        return

    selected_monitor = get_monitors()[monitor_index]

    # Load color chart template
    color_chart_template = cv2.imread('D:\\code\\image_validation\\Macbeth_Anode_ColorChart.tif')
    if color_chart_template is None:
        print(colored_info("Color chart template not found"))
        return

    # Capture screenshot from selected output
    print(colored_info(f"Capturing screenshot from output {monitor_index}"))
    image = capture_screenshot(selected_monitor)
    time.sleep(delay_time)

    # Calculate scaling factor
    scaling_factor = max(image.width / selected_monitor.width, image.height / selected_monitor.height)

    # Resize color chart template based on the scaling factor
    new_size = (int(color_chart_template.shape[1] * scaling_factor), int(color_chart_template.shape[0] * scaling_factor))
    color_chart_template = cv2.resize(color_chart_template, new_size)

    # Display color space and estimated image bit depth
    color_space, estimated_image_bit_depth = get_image_info(image)
    print(colored_info(f"Color space: {color_space}"))
    print(colored_info(f"Estimated image bit depth: {estimated_image_bit_depth}"))

    # Convert PIL image to OpenCV image
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Find color chart
    print(colored_info("Finding color chart, please wait..."))
    color_chart_coords = find_color_chart(opencv_image, color_chart_template)
    time.sleep(delay_time)

    # Validate color chart if found
    if color_chart_coords:
        print(colored_info("Validating color chart, please wait..."))
        validate_color_chart(opencv_image, color_chart_coords)
        time.sleep(delay_time)
        print(colored_info('Color chart found and validated'))
    else:
        print(colored_info('Color chart not found'))


if __name__ == '__main__':
    main()
