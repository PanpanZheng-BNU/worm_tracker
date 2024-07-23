import cv2, os
import numpy as np

# path to pic
p2img = "./pics/Naive2_1.jpg"
img = cv2.imread(p2img)

# path to video
p2video = "./Naive1.mp4"
ROIRadius = 900


def findROICenter(v):
    """
    using HoughCircles to find the center of the ROI
    :param v: the path to the video
    :return (cx, cy): the center of the ROI circle.
    """
    cap = cv2.VideoCapture(v)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_circles = cv2.HoughCircles(gray,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=30,
                                        param2=200, minRadius=1000, maxRadius=1150)
    a_avg, b_avg = 0, 0
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            a_avg += a
            b_avg += b

        a_avg = round(a_avg / detected_circles.shape[1])
        b_avg = round(b_avg / detected_circles.shape[1])
    return (a_avg, b_avg)


def selectROI(v, roiRadius):
    """
    selectROI(v, roiRadius): select the ROI from the video
    :param v: the path 2 video
    :param roiRadius: the radius of ROI
    :return:
    """
    cap = cv2.VideoCapture(v)
    roi_x, roi_y = findROICenter(v)
    object_detector = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.medianBlur(gray, 5)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
        div = np.float32(gray) / (close)
        res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
        res = cv2.medianBlur(res, 3)
        _, th2 = cv2.threshold(res, 220, 255, cv2.THRESH_TOZERO)
        # th2 = cv2.inRange(res, 10, 225)
        th3 = cv2.adaptiveThreshold(th2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 17)

        th4 = np.zeros_like(th3)
        th4 = cv2.circle(th4, (roi_x, roi_y), roiRadius, (255, 255, 255), -1)
        th4[th4 != 0] = th3[th4 != 0]
        # th4 = cv2.fastNlMeansDenoising(th4)
        th4 = cv2.medianBlur(th4, 3)
        th5 = object_detector.apply(frame)
        contours, _ = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            if 40 < cv2.contourArea(contour) < 150:
                print(cv2.arcLength(contour, True))
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # res = cv2.medianBlur()
        if ret:
            cv2.circle(frame, (roi_x, roi_y), roiRadius, (255, 0, 0), 3)
            cv2.imshow("th3", res)
            cv2.imshow("th2", th4)
            # cv2.imshow("th5", th5)
            # cv2.imshow("res", res)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break


if __name__ == '__main__':
    selectROI("./Naive1.mp4", 925)
