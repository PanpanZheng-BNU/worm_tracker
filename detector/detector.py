import multiprocessing
import os
import sys

import cv2
import numpy as np
import pandas as pd
import progressbar

from utils.detect_circle import *

sys.path.append("../")
# from conncat_videos  import find_all_videos


# def findROICenter(v):
#     """
#     findROICenter: using HoughCircles to find the center of the ROI
#     Args:
#         v: path2video
#     Returns:
#         (cx, cy): the center of the ROI circle.
#     """
#     cap = cv2.VideoCapture(v)
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     detected_circles = cv2.HoughCircles(gray,
#                                         cv2.HOUGH_GRADIENT, 1, 20, param1=30,
#                                         param2=200, minRadius=900, maxRadius=980)
#     if detected_circles is not None:
#         a_avg, b_avg = np.uint64(0), np.uint64(0)
#         detected_circles = np.uint16(np.around(detected_circles))
#         for pt in detected_circles[0, :]:
#             a, b, r = pt[0], pt[1], pt[2]
#             a_avg += a
#             b_avg += b
#
#         a_avg = round(a_avg / detected_circles.shape[1])
#         b_avg = round(b_avg / detected_circles.shape[1])
#     return (a_avg, b_avg)


def findROICenter(path2img):
    return detect_red_circles(path2img)


def detect(v_dict, p2s, roiRadius, vis, imgs, video, date):
    """
    detect(v_dict, roiRadius, vis, imgs, video,date): detect worms in the video
    Args:
        v_dict: path2video store in dict {subj: [path2videos...]}

    :param v_dict: the path 2 video
    :param roiRadius: the radius of ROI
    :param vis: whether to show the video
    :param imgs: whether to store the images
    :param video: whether to store the video
    :param date: the date of the experiment
    """

    # p2vs = os.path.dirname(v_dict[0][list(v_dict[0].keys())[0]][0])
    os.path.isdir(p2s) or os.makedirs(p2s)  # create the results folder

    vs = list(v_dict.values())[0]
    p2vs = os.path.dirname(vs[0])
    print(vs[0])
    subj_name = list(v_dict.keys())
    roi_x, roi_y = findROICenter(os.path.join(p2vs, subj_name[0] + ".jpg"))
    object_detector = cv2.createBackgroundSubtractorKNN(
        history=3500, dist2Threshold=80
        # history=1500, dist2Threshold=140
    )  # 1000 170
    tmp_cap = cv2.VideoCapture(vs[0])
    ret, frame = tmp_cap.read()
    fps = round(tmp_cap.get(cv2.CAP_PROP_FPS))
    p2r = os.path.join(p2s, subj_name[0] + "_" + date)
    os.path.isdir(p2r) or os.makedirs(p2r)

    p2rcsv = os.path.join(p2r, "csv")
    with open(os.path.join(p2r, "centroids.txt"), "w+") as f:
        f.write(f"[{roi_x}, {roi_y}]")
    os.path.isdir(p2rcsv) or os.makedirs(p2rcsv)

    if video:
        new_frame = cv2.VideoWriter(
            os.path.join(p2r, subj_name[0] + ".avi"),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (frame.shape[1], frame.shape[0]),
        )
    n_frame = 0
    frame_length = 0
    for v in vs:
        tmp_cap = cv2.VideoCapture(v)
        frame_length += int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bar = progressbar.ProgressBar(max_value=frame_length)
    mod_number = np.random.randint(0, 1000)
    print(mod_number)
    for v in vs:
        curr_v = cv2.VideoCapture(v)
        while curr_v.isOpened():
            df = pd.DataFrame(
                columns=[
                    "frame",
                    "x",
                    "y",
                    "w",
                    "h",
                    "cX",
                    "cY",
                    # "ellipse_w",
                    # "ellipse_h",
                ]
            )
            ret, frame = curr_v.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_gray = np.zeros_like(gray)
            # new_gray = cv2.circle(new_gray, (roi_x, roi_y),
            #                       roiRadius, (255, 255, 255), -1)
            new_gray = cv2.rectangle(
                new_gray,
                (roi_x - roiRadius, roi_y - roiRadius),
                (roi_x + roiRadius, roi_y + roiRadius),
                (255, 255, 255),
                -1,
            )
            new_gray[new_gray != 0] = gray[new_gray != 0]
            new_gray = object_detector.apply(new_gray)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_gray = cv2.morphologyEx(new_gray, cv2.MORPH_OPEN, kernel)
            new_gray = cv2.morphologyEx(new_gray, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                new_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            n_frame += 1

            for contour in contours:
                # if 20 < cv2.contourArea(contour) < 250:
                if 5 < cv2.contourArea(contour) < 250:
                    approx = cv2.approxPolyDP(
                        contour, 0.01 * cv2.arcLength(contour, True), True
                    )
                    x, y, w, h = cv2.boundingRect(approx)

                    # try:
                    # ellipse = cv2.fitEllipse(contour)
                    # ellipse_w, ellipse_h = ellipse[1]
                    # cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
                    # except:
                    # ellipse_w, ellipse_h = np.nan, ep.nan

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    M = cv2.moments(contour)
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    cv2.circle(frame, (int(cX), int(cY)), 5, (0, 0, 255), -1)
                    df.loc[len(df)] = [
                        n_frame,
                        x,
                        y,
                        w,
                        h,
                        cX,
                        cY,
                        # ellipse_w,
                        # ellipse_h,
                    ]

            # cv2.circle(frame, (roi_x, roi_y), roiRadius, (255, 0, 0), 3)
            cv2.rectangle(
                frame,
                (roi_x - roiRadius, roi_y - roiRadius),
                (roi_x + roiRadius, roi_y + roiRadius),
                (255, 0, 0),
                3,
            )

            df.to_csv(
                os.path.join(p2rcsv, "{}_frame{}.csv".format(subj_name[0], n_frame)),
                index=False,
            )
            bar.update(n_frame)

            if video:
                # new_frame.write(cv2.cvtColor(new_gray, cv2.COLOR_GRAY2BGR))
                new_frame.write(frame)

            if vis:
                cv2.imshow("detec", new_gray)
                cv2.imshow("frame", frame)
            if imgs:
                p2rimg = os.path.join(p2r, "images")
                os.path.isdir(p2rimg) or os.makedirs(p2rimg)
                cv2.imwrite(
                    os.path.join(
                        p2rimg, "{}_frame{}.jpg".format(subj_name[0], n_frame)
                    ),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 60],
                )

            (n_frame % 1000 == mod_number) and cv2.imwrite(
                os.path.join(p2r, "sample_img_{}.jpg".format(n_frame)),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 60],
            )
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    if vis:
        new_frame.release()


if __name__ == "__main__":
    # v = "../concat_video/N22.avi"
    os.path.isdir("csv4") or os.makedirs("csv4")
    v = "/Volumes/MyPassport/new_data/15cm/2023.7.14/"
    # print(find_all_videos(v)['W1165_naiv1'])
    new_dict = []
    for sub in find_all_videos(v).keys():
        new_dict.append({sub: find_all_videos(v)[sub]})
    print(len(new_dict))
    print([(i, j) for i, j in zip(new_dict, [1200 for _ in range(len(new_dict))])])
    p = multiprocessing.Pool()
    p.starmap(
        detect,
        [(i, j) for i, j in zip(new_dict, 1200 * np.ones(len(new_dict), dtype=int))],
    )

    # print(findROICenter(find_all_videos(v)['W1165_naiv1'][1]))
    # cap = cv2.VideoCapture(find_all_videos(v)['W1165_naiv3'][0])
    # print(find_all_videos(v))
    # print(find_all_videos(v).keys())
    # detect(v, 1200)
