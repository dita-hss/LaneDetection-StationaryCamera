import cv2
import numpy as np
from typing import Tuple


def gradient(image: np.ndarray) -> np.ndarray:
    """
    Applies gradient-based edge detection to an input image.

    Parameters:
        image (numpy.ndarray): Input image in RGB format.

    Returns:
        numpy.ndarray: Resulting image after applying edge detection.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5,5), 0)
    canny_img = cv2.Canny(blur_img, 50, 150)
    return canny_img

def region_of_interest(image: np.ndarray) -> np.ndarray:
    """
    Creates a region of interest mask on an input image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Mask representing the region of interest.
    """

    height = image.shape[0]
    # values are specific to the camera positioning
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Displays the given lines on the given image.

    Parameters:
        image (numpy.ndarray): Input image.
        lines (numpy.ndarray): Array of lines represented by (x1, y1, x2, y2).

    Returns:
        numpy.ndarray: Image with lines displayed.
    """
    line_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_img

def avg_slope_intercept(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Computes the average slope and intercept of lines and generates the coordinates of the left and right lines.

    Parameters:
        image (numpy.ndarray): Input image.
        lines (numpy.ndarray): Array of lines represented by (x1, y1, x2, y2).

    Returns:
        numpy.ndarray: Array of coordinates representing the left and right lines.
    """
    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        
        if left_fit:
            left_fit_avg = np.average(left_fit, axis=0)
            left_line = create_coords(image, left_fit_avg)
        else:
            print("No left lines detected.")
            left_line = np.array([0, 0, 0, 0])
        
        if right_fit:
            right_fit_avg = np.average(right_fit, axis=0)
            right_line = create_coords(image, right_fit_avg)
        else:
            print("No right lines detected.")
            right_line = np.array([0, 0, 0, 0])
        
    return np.array([right_line, left_line])

def create_coords(image: np.ndarray, line_parameters: Tuple[float, float]) -> np.ndarray:
    """
    Generates line coordinates based on the image and line parameters.

    Parameters:
        image (numpy.ndarray): Input image.
        line_parameters (tuple[float, float]): Parameters of the line (slope, intercept).

    Returns:
        numpy.ndarray: Array of coordinates representing the line.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


if __name__ == '__main__':
    image = cv2.imread('test_image.jpg')
    lane_img = np.copy(image)
    canny_img = gradient(lane_img)
    cropped_img = region_of_interest(canny_img)
    #edge detection
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = avg_slope_intercept(lane_img, lines)
    line_img = display_lines(lane_img, avg_lines)
    fin_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
    cv2.imshow("Result", fin_img)
    cv2.waitKey(0)


