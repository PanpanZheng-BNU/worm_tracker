import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

p2v = "/Volumes/MyPassport/new_data/线虫行为学/2025.4.12_N2_D1/MV-CA060-11GM (00J52746790)/N2_1.avi"

cap = cv2.VideoCapture(p2v)
ret, frame = cap.read()
# r = cv2.selectROI("select the area", frame)
# cropped_image = frame[int(r[1]):int(r[1]+r[3]),  
                    #   int(r[0]):int(r[0]+r[2])] 

# cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("./frame.png", frame)

new = cv2.imread("./frame.png")
img_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
t, new_image = cv2.threshold(img_gray, 10,240, cv2.THRESH_BINARY)
new_image = cv2.blur(new_image, (5, 5))
# result = new.copy()
# image = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)
# cv2.imshow("cropped_image", cropped_image)
# cv2.imshow("new_image", new_image)

# plt.imshow(new_image, "gray", vmin=0, vmax=255)
# cv2.imshow("frame", new)
# lower = np.array([155,25,0])
# upper = np.array([179,255,255])
# mask = cv2.inRange(image, lower, upper)
# result = cv2.bitwise_and(result, result, mask=mask)

# cv2.imshow('mask', mask)
# cv2.imshow('result', img_gray)



cv2.imshow('result', new_image)
detected_circles = cv2.HoughCircles(new_image,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 30, minRadius = 1, maxRadius = 25) 



df = pd.DataFrame(columns=['x','y'])



if detected_circles is not None: 
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
        df.loc[len(df)] = [a,b]
  
        # Draw the circumference of the circle. 
        cv2.circle(new, (a, b), r, (0, 255, 0), 2) 
  
        # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(new, (a, b), 1, (0, 0, 255), 3) 
# cv2.waitKey()

cv2.imshow("Detected Circle", new) 
cv2.waitKey(0)

print(df)
df["x"] = np.array(df["x"], dtype=np.int64)
df["y"] = np.array(df["y"], dtype=np.int64)
print(df.loc[1]- df.loc[0])

