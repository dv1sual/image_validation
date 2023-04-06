import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time
from termcolor import colored
from screeninfo import get_monitors
from colorama import init

init(autoreset=True)


def colored_info(text):
    return colored(f"[INFO] - {text}", "red")


def capture_screenshot(monitor):
    x, y, width, height = monitor.x, monitor.y, monitor.width, monitor.height
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


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
    # Implement your color validation logic here
    pass


def list_monitors():
    monitors = get_monitors()
    for i, monitor in enumerate(monitors):
        print(colored_info(f"Output {i}: {monitor}"))


def main():
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

    # Find color chart in the screenshot
    print(colored_info("Finding color chart, please wait..."))
    color_chart_coords = find_color_chart(image, color_chart_template)
    time.sleep(delay_time)

    # Validate color chart if found
    if color_chart_coords:
        print(colored_info("Validating color chart, please wait..."))
        validate_color_chart(image, color_chart_coords)
        time.sleep(delay_time)
        print(colored_info('Color chart found and validated'))
    else:
        print(colored_info('Color chart not found'))


if __name__ == '__main__':
    main()
