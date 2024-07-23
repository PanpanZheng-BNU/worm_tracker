import cv2, os
import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import product

p2f = "./Naive2.mp4"
cap = cv2.VideoCapture(p2f)
df = pd.DataFrame(columns=["x", "y"])
object_detector = cv2.createBackgroundSubtractorMOG2(history=1500)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(cap.get(cv2.CAP_PROP_FPS))

size = (frame_width, frame_height)
basicname = os.path.basename(p2f)

result = cv2.VideoWriter(os.path.join("./", basicname.split(".")[0] + "traced.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         5, size)
raw_columns = [["x{}".format(i), "y{}".format(i)] for i in range(13)]
df = pd.DataFrame(columns=sum(raw_columns, []))
while True:
    ret, frame = cap.read()

    if ret == True:
        # 1. Object Detection
        mask = object_detector.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = contours[0]

        num = 0
        xys = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

                M = cv2.moments(contour)
                if M['m00'] == 0:
                    break

                cx = int(M['m10'] // M['m00'])
                cy = int(M['m01'] // M['m00'])
                xys.extend([cx,cy])

                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
                num += 1
        if len(xys) < 26:
            xys = xys + [np.nan] * (2*13 - len(xys))
        elif len(xys) > 26:
            xys = xys[:26]

        df.loc[len(df)] = xys
        m_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", m_mask)

        result.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
df.to_csv("./t.csv",index=False)
