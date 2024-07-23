import cv2, os
import numpy as np
frame1 = cv2.imread("./pics/Naive2_0.jpg")
gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# gray1_blured = cv2.medianBlur(gray1, 1)
gray1_blured = gray1
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
close = cv2.morphologyEx(gray1_blured, cv2.MORPH_CLOSE, kernel1)
div = np.float32(gray1_blured) / (close)
res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
res = cv2.medianBlur(res, 3)
_, th2 = cv2.threshold(res, 225, 255, cv2.THRESH_TOZERO_INV)
# th2 = cv2.inRange(res, 10,225)
# th2 = cv2.medianBlur(th2,3)
th3 = cv2.adaptiveThreshold(th2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 21, 17)

cv2.imshow("subframe", gray1_blured)
cv2.imshow("res", res)
cv2.imshow("th2", th2)
cv2.imshow("th3", th3)
cv2.waitKey(0)
cv2.destroyAllWindows()