import os
import cv2
import pandas as pd
import numpy as np
import re


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, x1_max = x1, x1 + w1
    y1_min, y1_max = y1, y1 + h1
    x2_min, x2_max = x2, x2 + w2
    y2_min, y2_max = y2, y2 + h2

    x_min = max(x1_min, x2_min)
    x_max = min(x1_max, x2_max)
    y_min = max(y1_min, y2_min)
    y_max = min(y1_max, y2_max)

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def simple_iou_tracker(detections, t_min, sigma_iou=0.3):
    """
    simple IOU based tracker.
    Args:
        detections(list): list of detections per frame,
        sigma_l(float): low detection threshold
        sigma_h(float): high detection threshold
        sigma_iou(float): IOU threshold
        t_min (float): minimum track length in frames.

    Returns:
        list: list of trackes.
    """
    tracks_active = []
    tracks_finished = []
    indx = 0
    for frame_num, detection_frame in enumerate(detections, start=1):
        dets = df2list(detection_frame)

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['centroids'].append(best_match['centroid'])
                    track['end_frame'] = frame_num

                    updated_tracks.append(track)

                    del dets[[i for i, _ in enumerate(dets) if all(_['centroid'] == best_match['centroid'])][0]]
                    # dets.index(best_match)
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                if len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)
        new_tracks = [{'bboxes': [det['bbox']],
                       'centroids': [det['centroid']],
                       'start_frame': frame_num,
                       'end_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks
    tracks_finished += [track for track in tracks_active
                        if len(track['bboxes']) >= t_min]
    return tracks_finished


def find_csvs(p2result):
    """
    """
    re_num = r"frame(\d+).csv"
    for root, dirs, files in os.walk(p2result):
        csvs = [file for file in files if (".csv" in file) and file[0] != "."]
        csvs = [os.path.join(root, csv) for csv in csvs]
    csvs.sort(key=lambda x: int(re.findall(re_num, x)[0]))
    return csvs
if __name__ == "__main__":
    all_csvs = find_csvs("../results/N21")
    csv_list = []
    for csv in all_csvs:
        df = pd.read_csv(csv)
        csv_list.append(df)
    long_dfs = pd.concat(csv_list)
    long_dfs.to_csv("../results/N21.csv", index=False)

