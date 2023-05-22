from lanes_picture import gradient, region_of_interest, region_of_interest, avg_slope_intercept, display_lines
import cv2
import numpy as np

capture = cv2.VideoCapture("test_video.mp4")
while(capture.isOpened()):
    _, img_frame = capture.read()
    canny_img = gradient(img_frame)
    cropped_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = avg_slope_intercept(img_frame, lines)
    line_img = display_lines(img_frame, avg_lines)
    fin_img = cv2.addWeighted(img_frame, 0.8, line_img, 1, 1)
    cv2.imshow("Result", fin_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
    
