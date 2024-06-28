import cv2, os
import pandas as pd
import multiprocessing as mp
from itertools import product
# import numpy as np


# mypath = './'  # path to the folder containing the video files
# onlyfolders = [f for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]  # list of folders in the path
# path2v = onlyfolders[0]  # path to the folder containing the video files
# all_files = os.listdir(path2v)
# all_files.sort()
# path2f = os.path.join(path2v, all_files[4])


def tracing(*args):
    # print(args[0])
    cap = cv2.VideoCapture(args[0][0])
    df = pd.DataFrame(columns=['x', 'y'])
    object_detector = cv2.createBackgroundSubtractorKNN()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(cap.get(cv2.CAP_PROP_FPS))

    size = (frame_width, frame_height)
    fname = os.path.basename(args[0][0])
    print(os.path.basename(args[0][0]))

    result = cv2.VideoWriter(os.path.join(args[0][1], fname.split(".")[0] + ".mp4"),
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             10, size)

    while True:
        ret, frame = cap.read()

        if ret == True:
            # 1. Object Detection
            mask = object_detector.apply(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = contours[0]

            for contour in contours:
                if cv2.contourArea(contour) > cv2.contourArea(max_contour):
                    max_contour = contour

            contour = max_contour
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            M = cv2.moments(contour)
            if M['m00'] == 0:
                break

            cx = int(M['m10'] // M['m00'])
            cy = int(M['m01'] // M['m00'])

            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
            df.loc[len(df)] = [cx, cy]
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
    df.to_excel(fname.split(".")[0] + ".xlsx", index=False)


if __name__ == '__main__':
    p2s = './all_results'
    if not os.path.isdir(p2s):
        os.makedirs(p2s)
    # two_files = [os.path.join(path2v, all_files[0]),
    #              # os.path.join(path2v, all_files[1]),
    #              os.path.join(path2v, all_files[2])]
    # with mp.Pool(processes=2) as pool:
    #     pool.starmap(tracing, product(zip(two_files, [p2s] * len(two_files))))

    tracing(['p2con/N2_Naive1.avi',p2s])

    # for i in zip(two_files, [p2s] * len(two_files)):
    #     print(*i)

    # tracing(path2f)
    # capture_process.start()
