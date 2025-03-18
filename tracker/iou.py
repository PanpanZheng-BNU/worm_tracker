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


# def simple_iou_tracker(detections, t_min, sigma_iou=0.3):
#     """
#     simple IOU based tracker.
#     Args:
#         detections(list): list of detections per frame,
#         sigma_l(float): low detection threshold
#         sigma_h(float): high detection threshold
#         sigma_iou(float): IOU threshold
#         t_min (float): minimum track length in frames.

#     Returns:
#         list: list of trackes.
#     """
#     tracks_active = []
#     tracks_finished = []
#     indx = 0
#     for frame_num, detection_frame in enumerate(detections, start=1):
#         dets = df2list(detection_frame)

#         updated_tracks = []
#         for track in tracks_active:
#             if len(dets) > 0:
#                 best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
#                 if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
#                     track['bboxes'].append(best_match['bbox'])
#                     track['centroids'].append(best_match['centroid'])
#                     track['end_frame'] = frame_num

#                     updated_tracks.append(track)

#                     del dets[[i for i, _ in enumerate(dets) if all(_['centroid'] == best_match['centroid'])][0]]
#                     # dets.index(best_match)
#             if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
#                 if len(track['bboxes']) >= t_min:
#                     tracks_finished.append(track)
#         new_tracks = [{'bboxes': [det['bbox']],
#                        'centroids': [det['centroid']],
#                        'start_frame': frame_num,
#                        'end_frame': frame_num} for det in dets]
#         tracks_active = updated_tracks + new_tracks
#     tracks_finished += [track for track in tracks_active
#                         if len(track['bboxes']) >= t_min]
#     return tracks_finished



def df2list(df):
    """
    convert pandas' df into a list consist of dict.
    Args:
        df: dataframe
    Returns:
        dets: a list
    """
    dets = [{'bbox':row[['x','y','w','h']].values, 'centroid':row[['cX','cY']].values, "oval": row[["ellipse_w", "ellipse_h"]]} for _, row in df.iterrows()]
    return dets

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
                    track['ovals'].append(best_match['oval'])
                    track['end_frame'] = frame_num

                    updated_tracks.append(track)

                    
                    del dets[[i for i, _ in enumerate(dets) if all(_['centroid'] == best_match['centroid'])][0]]
                    # dets.index(best_match)
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                if len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)
        new_tracks = [{'bboxes': [det['bbox']],
                       'centroids': [det['centroid']], 
                          'ovals': [det['oval']],
                       'start_frame': frame_num,
                       'end_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks
    tracks_finished += [track for track in tracks_active
                        if len(track['bboxes']) >= t_min]
    return tracks_finished

# def Repeat(x):
#     _size = len(x)
#     repeated = []
#     for i in range(_size):
#         k = i + 1
#         for j in range(k, _size):
#             if x[i] == x[j] and x[i] not in repeated:
#                 repeated.append(x[i])
#     return repeated
                
def bboxes2indx(df, trackers):
    """
    Args:
        df: dataframe
        trackers: list of trackers
    Returns:
        indx: list of index
    """
    indx = []
    for i, track in enumerate(trackers):
        find_all = np.all(np.array(track['bboxes']) == np.array(df[["x", "y", "w", "h"]]), axis=1)
        is_exists = np.any(find_all)
        the_frame = np.where(find_all == True)[0] + track['start_frame']
        # if is_exists and the_frame == df['frame'].values:
        if is_exists and the_frame == df['frame'].values:
            indx.append(i)
    return np.array(indx)

def df_iou(last_bbox,df):
    """
    df_iou: Calculating the IOU between one bboxes and all bboxes in a dataframe. And return the bboxes with IOU > 0.
    Args:
        last_bbox: a bbox
        df: a frame dataframe from all_dfs
    """
    return df[([iou(last_bbox, i) > 0 for i in np.array(df[["x" ,"y" ,"w" ,"h"]])])]

def find_initial(all_dfs, num, end_frame):
    """
    Args:
        all_dfs: list of dataframes
        num: int
    Returns:
        indx: list of index
    """
    len_df = np.array([len(df) for df in all_dfs[:end_frame-1]])
    indx_target = np.where(len_df >= num)[0]
    threshold = 1
    out = np.array_split(indx_target, np.flatnonzero(np.diff(indx_target) > threshold) + 1)

    return out[np.argmax([len(i) for i in out])]
