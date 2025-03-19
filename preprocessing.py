import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import re
from tracker.iou import *
import argparse


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Process some integers.')
    parse.add_argument('--date', '-d', type=str, help='date of the experiment')
    parse.add_argument('--p2s', '-p', type=str, help='path to store the preprocessed data', default="data")
    args = parse.parse_args()
    os.path.isdir(args.p2s) or os.makedirs(args.p2s)
    
    csv_dicts = dict()
    match_num = r"(\d+).csv"
    for root, dirs, files in os.walk("results"):
        tmp_files = []
        for file in files:
            if file.endswith(".csv") and (args.date in root):
                tmp_files.append(file)
        if len(tmp_files):
            csv_dicts[root.split(os.sep)[1]] = [root,tmp_files]

    for k in csv_dicts.keys():
        csv_dicts[k][1].sort(key = lambda x: int(re.findall(match_num, x)[0]))
        subj = csv_dicts[k]
        all_dfs = [pd.read_csv(os.path.join(subj[0], i)) for i in subj[1]]
        all_trackers = simple_iou_tracker(all_dfs,10, 0.3)
        all_subjects_dict = {}
        for i, tracker in enumerate(all_trackers):
            all_subjects_dict[i] = {"start_frame": tracker['start_frame'], "end_frame": tracker['end_frame'], "bboxes": np.array(tracker['bboxes']).tolist(), "centroids": np.array(tracker['centroids']).tolist(),
                                    "ovals": np.array(tracker['ovals']).tolist()}
        with open(os.path.join(args.p2s,'./{}_trackers.json'.format(k)), 'w') as f:
            json.dump(all_subjects_dict, f)
        pd.concat(all_dfs, ignore_index=True).to_csv(os.path.join(args.p2s, './{}_dfs.csv'.format(k)), index=False)