import sys
sys.path.append('..')
import pandas as pd
import numpy as np


# iou function, calculating the iou between two boxes
def iou(box1, box2):
    """
    iou: Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1: (x1, y1, w1, h1) coordinates of the first bounding box
        box2: (x2, y2, w2, h2) coordinates of the second bounding box
    """
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


def df2list(df):
    """
    convert pandas' df into a list consist of dict for simple IoU tracking
    Args:
        df: dataframe
    Returns:
        dets: a list consists of dictionaries
    """
    dets = [{'bbox':row[['x','y','w','h']].values, 
             'centroid':row[['cX','cY']].values, 
             "oval": row[["ellipse_w", "ellipse_h"]]} for _, row in df.iterrows()]
    return dets

def simple_iou_tracker(detections, t_min, sigma_iou=0.3):
    """
    simple IOU based tracker.
    Args:
        detections(list): list of detections per frame, 
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



# def df_iou(last_bbox,df):
#     """
#     df_iou: Calculating the IOU between one bboxes and all bboxes in a frame. And return the bboxes with IOU > 0.
#     Args:
#         last_bbox: a bbox
#         df: a frame dataframe from all_dfs
#     """
#     return df[([iou(last_bbox, i) > 0 for i in np.array(df[["x" ,"y" ,"w" ,"h"]])])]


# def find_initial(long_dfs, num, trackers_summarize):
#     """
#     Args:
#         long_dfs: list of dataframes
#         num: int
#     Returns:
#         indx: list of index
#     """
#     counts_df = long_dfs.groupby("frame").size()
#     counts_df.index = counts_df.index.astype(int)
#     counts_df.columns = ['counts']

#     indx_target = counts_df[counts_df >= num].index
#     threshold = 1
#     indx_split = np.split(indx_target, np.where(np.diff(indx_target) > threshold)[0] + 1)
#     arg_max_split = np.argmax([len(i) for i in indx_split])
#     frame_min, frame_max = indx_split[arg_max_split][0], indx_split[arg_max_split][-1]
#     tmp_df = trackers_summarize.loc[(trackers_summarize['start_frame'] <= frame_min) & (trackers_summarize['end_frame'] >= frame_max)]

#     return tmp_df

# def generate_summarize(all_trackers):
#     """
#     Args:
#         all_trackers: dict of trackers
#     Returns:
#         tmp_df: dataframe summarizing all trackers, consists of num, start_frame, end_frame, durations, start_bbox, end_bbox, start_centroids, end_centroids
#     """
#     tmp_df = pd.DataFrame(columns=["num", "start_frame", "end_frame", "durations", "start_bbox", "end_bbox", "start_centroids", "end_centroids"])
#     for i in all_trackers:
#         tmp_num = int(i)
#         tmp_tracker = all_trackers[i]
#         tmp_start_frame = tmp_tracker['start_frame']
#         tmp_end_frame = tmp_tracker['end_frame']
#         tmp_durations = tmp_end_frame - tmp_start_frame
#         tmp_start_bbox = tmp_tracker['bboxes'][0]
#         tmp_end_bbox = tmp_tracker['bboxes'][-1]
#         tmp_start_centroids = tmp_tracker['centroids'][0]
#         tmp_end_centroids = tmp_tracker['centroids'][-1]
#         tmp_df.loc[len(tmp_df)] = [tmp_num, tmp_start_frame, tmp_end_frame, tmp_durations, tmp_start_bbox, tmp_end_bbox, tmp_start_centroids, tmp_end_centroids]
#     return tmp_df


# def find_bbox(all_trackers, bbox_df):
#     tmp_indx = []
#     tmp_frame = []
#     for j in range(len(bbox_df)):
#         bbox = bbox_df.iloc[j][["x", "y", "w", "h"]].values
#         for i in all_trackers:
#             indx = np.where(np.all(np.array(all_trackers[i]['bboxes']) == bbox, axis=1))[0]
#             for k in indx:
#                 if (k + all_trackers[i]['start_frame']) in bbox_df.index:
#                     tmp_indx.append(int(i))
#                     tmp_frame.append(bbox_df.index.values[0])
#     return tmp_indx, tmp_frame

# def right_find(tmp_df, long_dfs):
#     """
#     Args:
#         trackers_summarize: dataframe summarizing all trackers, consists of num, start_frame, end_frame, durations, start_bbox, end_bbox, start_centroids, end_centroids
#         long_dfs: the information of each frame in a video
#     Returns:
#     """

#     tmp_end_bbox = tmp_df.end_bbox
#     tmp_end_frame = tmp_df.end_frame

#     return df_iou(tmp_end_bbox,long_dfs.loc[[tmp_end_frame + 1]])

# def left_find(tmp_df, long_dfs):
#     """
#     Args:
#         trackers_summarize: dataframe summarizing all trackers, consists of num, start_frame, end_frame, durations, start_bbox, end_bbox, start_centroids, end_centroids
#         long_dfs: the information of each frame in a video
#     Returns:
#     """

#     tmp_start_bbox = tmp_df.start_bbox
#     tmp_start_frame = tmp_df.start_frame

#     return df_iou(tmp_start_bbox,long_dfs.loc[[tmp_start_frame - 1]])

# def edge_type(tmp_df, long_dfs, all_trackers):
#     """
#     Args:
#         trackers_summarize: dataframe summarizing all trackers, consists of num, start_frame, end_frame, durations, start_bbox, end_bbox, start_centroids, end_centroids
#         long_dfs: the information of each frame in a video
#     Returns:
#     """

#     if tmp_df.start_frame-1 not in long_dfs.index:
#         left_prev = ([], [])
#     else:
#         left_find_df = left_find(tmp_df, long_dfs)
#         left_prev = find_bbox(all_trackers, left_find_df)

#     if tmp_df.end_frame+1 not in long_dfs.index:
#         right_next = ([], [])
#     else:
#         right_find_df = right_find(tmp_df, long_dfs)
#         right_next = find_bbox(all_trackers, right_find_df)

#     if len(right_next[0]) == 0:
#         tmp_right_type = "disappear"
#     elif len(right_next[0]) == 1:
#         tmp_right_type = "merge"
#     elif len(right_next[0]) > 1:
#         tmp_right_type = "split"

#     if len(left_prev[0]) == 0:
#         tmp_left_type = "appear"
#     elif len(left_prev[0]) == 1:
#         tmp_left_type = "split"
#     elif len(left_prev[0]) > 1:
#         tmp_left_type = "merge"


#     return tmp_left_type, tmp_right_type, left_prev, right_next


# # tmp methods for conn_disappear
# def conn_disappear_next(tmp_df, merge_trackers_summarize):
#     tmp_end_frame = tmp_df.end_frame
#     tmp_end_centroids = tmp_df.end_centroids
#     tmp_end_bbox = tmp_df.end_bbox

#     indx_frame = (merge_trackers_summarize.start_frame > tmp_end_frame) & ((merge_trackers_summarize.start_frame - tmp_end_frame) <= 100) & (merge_trackers_summarize.left_type == "appear")

#     tmp_next_df = merge_trackers_summarize.loc[indx_frame]
#     # print(tmp_next_df)
#     if len(tmp_next_df) == 0:
#         return "can't find"
#     else:
#         distance_series = pd.Series({i: np.linalg.norm(np.array(tmp_next_df.loc[i].start_centroids) - np.array(tmp_end_centroids)) for i in tmp_next_df.index}, name="distance")
#         diff_frames_series = pd.Series({i: np.abs(tmp_next_df.loc[i].start_frame - tmp_end_frame) for i in tmp_next_df.index}, name="diff_frames")
#         overlapping_series = pd.Series({i: iou(tmp_next_df.loc[i].start_bbox, tmp_end_bbox) > 0 for i in tmp_next_df.index}, name="is_overlapping")
#         tmp_df = pd.concat([tmp_next_df, overlapping_series, diff_frames_series, distance_series], axis=1)
#         tmp_df = tmp_df.loc[tmp_df.distance / tmp_df.diff_frames < 5]
#         if len(tmp_df) == 0:
#             return "can't find"
#         tmp_df.sort_values(by=["distance", "diff_frames"], inplace=True)
#         return tmp_df.iloc[0].num

# def conn_disappear_prev(tmp_df, merge_trackers_summarize):
#     tmp_start_frame = tmp_df.start_frame
#     tmp_start_centroids = tmp_df.start_centroids
#     tmp_start_bbox = tmp_df.start_bbox

#     indx_frame = (merge_trackers_summarize.end_frame < tmp_start_frame) & (np.abs(merge_trackers_summarize.end_frame - tmp_start_frame) <= 100) & (merge_trackers_summarize.right_type == "disappear")

#     tmp_prev_df = merge_trackers_summarize.loc[indx_frame]
#     tmp_indx = tmp_prev_df.index.to_numpy()
#     # print(tmp_prev_df)
#     if len(tmp_prev_df) == 0:
#         return "can't find"
#     else:
#         distance_series = pd.Series({i: np.linalg.norm(np.array(tmp_prev_df.loc[i].end_centroids) - np.array(tmp_start_centroids)) for i in tmp_indx}, name="distance")
#         diff_frames_series = pd.Series({i: np.abs(tmp_prev_df.loc[i].end_frame - tmp_start_frame) for i in tmp_indx}, name="diff_frames")
#         overlapping_series = pd.Series({i: iou(tmp_prev_df.loc[i].end_bbox, tmp_start_bbox) > 0 for i in tmp_indx}, name="is_overlapping")
#         mean_velocity_series = pd.Series({i: np.linalg.norm(np.array(tmp_prev_df.loc[i].end_centroids) - np.array(tmp_start_centroids)) / np.abs(tmp_prev_df.loc[i].end_frame - tmp_start_frame) < 5 for i in tmp_indx}, name="mean_velocity")
#         tmp_df = pd.concat([tmp_prev_df, overlapping_series, diff_frames_series,mean_velocity_series, distance_series], axis=1)
#         tmp_df = tmp_df.loc[tmp_df.distance / tmp_df.diff_frames < 5]
#         if len(tmp_df) == 0:
#             return "can't find"
#         tmp_df.sort_values(by=["distance", "diff_frames"], inplace=True)
#         return tmp_df.iloc[0].num

# def split_from(num, merge_trackers_summarize):
#     tmp_df = merge_trackers_summarize
#     tmp_df_1 = tmp_df.loc[[num in j[0] for j in tmp_df.left_prev]]

#     this_end = tmp_df.loc[tmp_df.num == num].end_frame.values
#     this_start = tmp_df.loc[tmp_df.num == num].start_frame.values

#     result_df = pd.DataFrame(columns=["this_num", "nex_num", "this_start", "this_end", "nex_start", "nex_end"])
#     result_df.loc[len(result_df)] = [num, tmp_df_1.num.values, this_start, this_end, tmp_df_1.start_frame.values, tmp_df_1.end_frame.values]
#     result_df['difference'] = np.abs(result_df.nex_start - result_df.this_end) - 1
#     result_df['is_split'] = np.any(result_df['difference'].values[0])

#     return result_df

# def merge_to(num, merge_new_summarize):
#     tmp_df = merge_new_summarize
#     tmp_df_1 = tmp_df.loc[[num in j[0] for j in tmp_df.right_next]]

#     this_end = tmp_df.loc[tmp_df.num == num].end_frame.values
#     this_start = tmp_df.loc[tmp_df.num == num].start_frame.values

#     result_df = pd.DataFrame(columns=["this_num", "nex_num", "this_start", "this_end", "nex_start", "nex_end"])
#     result_df.loc[len(result_df)] = [num, tmp_df_1.num.values, this_start, this_end, tmp_df_1.start_frame.values, tmp_df_1.end_frame.values]
#     result_df['difference'] = np.abs(result_df.nex_end - result_df.this_start) - 1
#     result_df['is_merge'] = np.any(result_df['difference'].values[0])
#     return result_df

# def split_tracker(tracker, frame):
#     """
#     Args:
#         tracker: dict of tracker
#         frame: int
#     Returns:
#         new_tracker: dict of tracker
#     """
#     new_tracker1 = {}       
#     new_tracker1['bboxes'] = tracker['bboxes'][:(frame - tracker['start_frame'] + 1)]
#     new_tracker1['centroids'] = tracker['centroids'][:(frame - tracker['start_frame'] + 1)]
#     new_tracker1['ovals'] = tracker['ovals'][:(frame - tracker['start_frame'] + 1)]
#     new_tracker1['start_frame'] = tracker['start_frame']
#     new_tracker1['end_frame'] = frame
#     new_tracker2 = {}
#     new_tracker2['bboxes'] = tracker['bboxes'][(frame - tracker['start_frame'] + 1):]
#     new_tracker2['centroids'] = tracker['centroids'][(frame - tracker['start_frame']+1):]
#     new_tracker2['ovals'] = tracker['ovals'][(frame - tracker['start_frame']+1):]
#     new_tracker2['start_frame'] = frame + 1
#     new_tracker2['end_frame'] = tracker['end_frame']
#     return new_tracker1, new_tracker2



# def split_trackers(this_num, all_critical_split_df, all_trackers):
#     sorted_df = all_critical_split_df.loc[all_critical_split_df.tracker_num == this_num].sort_values('split_frame')
#     this_tracker = all_trackers[this_num]

#     tmp_str = []
#     for i in range(len(sorted_df)):
#         split_frame = sorted_df.iloc[i].split_frame - 1
#         t1, t2 = split_tracker(this_tracker, split_frame)
#         this_tracker = split_tracker(this_tracker, split_frame)[1]

#         if i < len(sorted_df) - 1:
#             tmp_str.append(t1)
#         else:
#             tmp_str.append(t1)
#             tmp_str.append(t2)
#     return tmp_str


# def find_next(num, merge_summary_df):
#     """
#     Args:
#         num: int
#         merge_summary_df: dataframe
#     Returns:
#         next_num: int
#     """
#     tmp_df = merge_summary_df.loc[num]
#     result_df = pd.DataFrame(columns=['this_num', 'nex_num', 'right_type'])
#     if tmp_df.right_type == "disappear":
#         result_df.loc[len(result_df)] = [num, conn_disappear_next(tmp_df, merge_summary_df), "disappear"]
#     else:
#         for i in tmp_df.right_next[0]:
#             result_df.loc[len(result_df)] = [num, i, tmp_df.right_type]
#     return result_df

# def find_all_nex(start_num, merge_summary_df, store_dict = [], banned_list = []):
#     find_next_df = find_next(start_num, merge_summary_df)
#     store_dict.append(find_next_df)
#     # print("${} find next: {}".format(start_num, find_next_df.nex_num.values))
#     # print("$is in banned list: {}".format([i in banned_list for i in find_next_df.nex_num.values]))
#     if np.all([i in banned_list for i in find_next_df.nex_num.values]):
#         print("${} all path banned".format(find_next_df.nex_num.values))
#         store_dict.append("all_path_banned")
#         return "all_path_banned"
#     if type(find_next_df.nex_num.values[0]) != str:
#         rand_int = np.random.randint(0, len(find_next_df))
#         while find_next_df.nex_num.values[rand_int] in banned_list:
#             rand_int = np.random.randint(0, len(find_next_df))
#         find_all_nex(find_next_df.nex_num.values[rand_int], merge_summary_df, store_dict, banned_list)

# def simple_find_nex(start_num, merge_df, banned_list):
#     multi_index = merge_df.loc[(merge_df.right_type == "split") | (merge_df.left_type == "merge")].index.values
#     tmp_list = []
#     find_all_nex(start_num, merge_df, tmp_list, banned_list)
#     t_num = [i.this_num.values for i in tmp_list if type(i) != str]
#     tmp_num = np.concatenate(t_num)
#     indexes = np.unique(tmp_num, return_index=True)[1]
#     worm = np.array([tmp_num[i] for i in sorted(indexes)])
#     new_banned_list = worm[np.where([i not in multi_index for i in worm])[0]]
#     new_banned_list = np.unique(np.concatenate([banned_list, new_banned_list]))
#     if type(tmp_list[-1]) == str:
#         worm = np.concatenate([worm, [-1]] )

#     return worm, new_banned_list
        
#     # tmp_start_frame = tmp_df.start_frame
#     # tmp_start_centroids = tmp_df.start_centroids
#     # tmp_start_bbox = tmp_df.start_bbox

#     # indx_frame = (merge_trackers_summarize.end_frame < tmp_start_frame) & (np.abs(merge_trackers_summarize.end_frame - tmp_start_frame) <= 100) & (merge_trackers_summarize.right_type == "disappear")

#     # tmp_prev_df = merge_trackers_summarize.loc[indx_frame]
#     # if len(tmp_prev_df) == 0:
#     #     return "can't find"
#     # else:
#     #     for i in range(len(tmp_prev_df)):
#     #         if iou(tmp_prev_df.iloc[i].end_bbox, tmp_start_bbox) > 0:
#     #             return tmp_prev_df.iloc[i].num
#     #         if np.linalg.norm(np.array(tmp_prev_df.iloc[i].start_centroids) - np.array(tmp_start_centroids)) / np.abs(tmp_prev_df.iloc[i].start_frame - tmp_start_frame) < 5:
#     #             return tmp_prev_df.iloc[i].num
#     #     return "can't find"

# def find_prev(num, merge_summary_df):
#     """
#     Args:
#         num: int
#         merge_summary_df: dataframe
#     Returns:
#         next_num: int
#     """
#     tmp_df = merge_summary_df.loc[num]
#     result_df = pd.DataFrame(columns=['this_num', 'prev_num', 'left_type'])
#     if tmp_df.left_type == "appear":
#         result_df.loc[len(result_df)] = [num, conn_disappear_prev(tmp_df, merge_summary_df), "appear"]
#     else:
#         for i in tmp_df.left_prev[0]:
#             result_df.loc[len(result_df)] = [num, i, tmp_df.left_type]
#     return result_df

# def find_all_prev(start_num, merge_summary_df, store_dict = [], banned_list = []):
#     find_prev_df = find_prev(start_num, merge_summary_df)
#     store_dict.insert(0, find_prev_df)
#     # print("${} find next: {}".format(start_num, find_next_df.nex_num.values))
#     # print("$is in banned list: {}".format([i in banned_list for i in find_next_df.nex_num.values]))
#     if np.all([i in banned_list for i in find_prev_df.prev_num.values]):
#         print("${} all path banned".format(find_prev_df.prev_num.values))
#         store_dict.insert(0,"all_path_banned")
#         return "all_path_banned"
#     if type(find_prev_df.prev_num.values[0]) != str:
#         rand_int = np.random.randint(0, len(find_prev_df))
#         while find_prev_df.prev_num.values[rand_int] in banned_list:
#             rand_int = np.random.randint(0, len(find_prev_df))
#         find_all_prev(find_prev_df.prev_num.values[rand_int], merge_summary_df, store_dict, banned_list)


# def simple_find_prev(start_num, merge_df, banned_list):
#     multi_index = merge_df.loc[(merge_df.right_type == "split") | (merge_df.left_type == "merge")].index.values
#     tmp_list = []
#     find_all_prev(start_num, merge_df, tmp_list, banned_list)
#     t_num = [i.this_num.values for i in tmp_list if type(i) != str]
#     tmp_num = np.concatenate(t_num)
#     indexes = np.unique(tmp_num, return_index=True)[1]
#     worm = np.array([tmp_num[i] for i in sorted(indexes)])
#     new_banned_list = worm[np.where([i not in multi_index for i in worm])[0]]
#     new_banned_list = np.unique(np.concatenate([banned_list, new_banned_list]))
#     if type(tmp_list[0]) == str:
#         worm = np.concatenate([[-1], worm])

#     return worm, new_banned_list
        