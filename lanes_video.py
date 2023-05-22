import lanes_picture
import cv2
import numpy as np

#MUST ALSO SET UP CAMERA POSITION IN REGION OF INTEREST
capture = cv2.VideoCapture("PATH TO VIDEO")
while(capture.isOpened()):
    _, img_frame = capture.read()
    canny_img = lanes_picture.gradient(img_frame)
    cropped_img = lanes_picture.region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 30, np.array([]), minLineLength=40, maxLineGap=5)
    avg_lines = lanes_picture.avg_slope_intercept(img_frame, lines)
    line_img = lanes_picture.display_lines(img_frame, avg_lines)
    fin_img = cv2.addWeighted(img_frame, 0.8, line_img, 1, 1)
    cv2.imshow("Result", fin_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
    
