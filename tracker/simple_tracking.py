import json
import pandas as pd
import numpy as np
import shutil
import re
from tracker.iou import *
import argparse
import os
import sys
sys.path.append("../")


def mv_centroids(args):
    centroids_dict = {}
    for root, dirs, files in os.walk(args.p2det):
        tmp_files = []
        for file in files:
            if file == "centroids.txt" and (args.date in root):
                tmp_files.append(file)
        if len(tmp_files):
            centroids_dict[root.split(os.sep)[-1]] = [root,tmp_files]
    print(centroids_dict)
    for k, i in centroids_dict.items():
        if not os.path.isdir(os.path.join(args.p2trackers, k.lower())):
            os.makedirs(os.path.join(args.p2trackers, k.lower()))
        shutil.copy(os.path.join(i[0], i[1][0]), os.path.join(args.p2trackers, k.lower(), "centroids.txt"))

def generate_trackers_and_long_df(args):
    match_num = r"(\d+).csv"
    csv_dicts = dict()
    for root, dirs, files in os.walk(args.p2det):
        tmp_files = []
        for file in files:
            if file.endswith(".csv") and (args.date in root):
                tmp_files.append(file)
        if len(tmp_files):
            csv_dicts[root.split(os.sep)[-2]] = [root,tmp_files]

    
    
    for k in csv_dicts.keys():
        if not os.path.isdir(os.path.join(args.p2trackers, k.lower())):
            os.makedirs(os.path.join(args.p2trackers, k.lower()))

        csv_dicts[k][1].sort(key = lambda x: int(re.findall(match_num, x)[0]))
        subj = csv_dicts[k]
        all_dfs = [pd.read_csv(os.path.join(subj[0], i)) for i in subj[1]]
        all_trackers = simple_iou_tracker(all_dfs,10, 0.3)
        all_subjects_dict = {}
        for i, tracker in enumerate(all_trackers):
            all_subjects_dict[i] = {"start_frame": tracker['start_frame'], "end_frame": tracker['end_frame'], "bboxes": np.array(tracker['bboxes']).tolist(), "centroids": np.array(tracker['centroids']).tolist(),
                                    "ovals": np.array(tracker['ovals']).tolist()}
        with open(os.path.join(args.p2trackers,k.lower(),'trackers.json'), 'w') as f:
            json.dump(all_subjects_dict, f)
        pd.concat([i for i in all_dfs if not i.empty], ignore_index=True).to_csv(os.path.join(args.p2trackers, k.lower(),'long_dfs.csv'), index=False)
        print(k, "done")
    


if __name__ == "__main__":
    # parse = argparse.ArgumentParser(description='Process some integers.')
    # parse.add_argument('--date', '-d', type=str, help='date of the experiment')
    # parse.add_argument('--p2s', '-p', type=str, help='path to store the preprocessed data', default="data")
    # args = parse.parse_args()
    mv_centroids(args)
    generate_trackers_and_long_df(args)
    