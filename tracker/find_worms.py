import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import re
import warnings
import os

# from iou import *
from plotly.tools import mpl_to_plotly
import plotly.io as pio
from matplotlib import pyplot as plt
from tracker.iou import *
from tracker.concat_trackers import *


def load_subj(p2trackers, subj):
    """
    Load a subject's data from folder. The subject's data is stored in a JSON file, a csv file, and a txt file
    """

    p2json = os.path.join(p2trackers, subj.lower(), "trackers.json")
    p2centroids = os.path.join(p2trackers, subj.lower(), "centroids.txt")
    with open(p2json, 'r') as f:
        subj_trackers = json.load(f)

    with open(p2centroids, 'r') as f:
        subj_centroids = np.array(eval(f.readline()))
    subj_long_dfs = pd.read_csv(os.path.join(
        p2trackers, subj.lower(), "long_dfs.csv"), index_col="frame")
    subj_long_dfs.index = subj_long_dfs.index.astype(int)

    subj_trackers = {int(k): v for k, v in subj_trackers.items()}

    return subj_trackers, subj_long_dfs, subj_centroids


def generate_merge_summarize(trackers, dfs):
    tmp_summarize = generate_summarize(trackers)
    tmp_trackers_type_df = pd.DataFrame(
        columns=["tracker_id", "left_type", "right_type", "left_prev", "right_next"])
    for i in range(len(tmp_summarize)):
        tmp_df = tmp_summarize.iloc[i]
        tmp_left_type, tmp_right_type, left_prev, right_next = edge_type(
            tmp_df, dfs, trackers)
        tmp_trackers_type_df.loc[len(tmp_trackers_type_df)] = [
            tmp_df.tracker_id, tmp_left_type, tmp_right_type, left_prev, right_next]

    tmp_merged_summarize = pd.merge(
        tmp_summarize, tmp_trackers_type_df, on="tracker_id")
    return tmp_merged_summarize


def trackers2fine(trackers, dfs):
    tmp_merge_summarize = generate_merge_summarize(trackers, dfs)
    split_from_df = pd.concat([split_from(i, tmp_merge_summarize) for i in range(
        len(tmp_merge_summarize))], ignore_index=True)
    split_from_df = split_from_df.loc[split_from_df.is_split == True]
    tmp_all_id = []
    for i in range(len(split_from_df)):
        nex_id = split_from_df.iloc[i].nex_id
        diff = split_from_df.iloc[i]['difference']
        nex_start = split_from_df.iloc[i]["nex_start"]
        this_id = split_from_df.iloc[i].this_id
        tmp_df = pd.DataFrame(columns=['tracker_id', 'split_frame'])
        for i, j in zip(nex_start, diff):
            if j != 0:
                tmp_df.loc[len(tmp_df)] = [this_id, i]
        tmp_all_id.append(tmp_df)
    # print(tmp_all_num[0])
    if len(tmp_all_id) == 0:
        tmp_new_trackers = trackers
    elif len(tmp_all_id) >= 1:
        critical_split_df = pd.concat(tmp_all_id, ignore_index=True)
        tmp_new_trackers = {}
        ini_indx = 0
        for i in range(len(trackers)):
            if i not in critical_split_df.tracker_id.values:
                tmp_new_trackers[ini_indx] = trackers[i]
                ini_indx += 1
            else:
                split_result = split_trackers(i, critical_split_df, trackers)
                for j in split_result:
                    tmp_new_trackers[ini_indx] = j
                    ini_indx += 1

    tmp_new_summarize = generate_merge_summarize(tmp_new_trackers, dfs)

    merge_to_df = pd.concat([merge_to(i, tmp_new_summarize)
                            for i in range(len(tmp_new_summarize))], ignore_index=True)
    merge_to_df = merge_to_df.loc[merge_to_df.is_merge == True]
    tmp_all_id = []
    for i in range(len(merge_to_df)):
        diff = merge_to_df.iloc[i]['difference']
        nex_start = merge_to_df.iloc[i]["nex_end"]
        this_id = merge_to_df.iloc[i].this_id

        tmp_df = pd.DataFrame(columns=['tracker_id', 'split_frame'])

        for i, j in zip(nex_start, diff):
            if j != 0:
                tmp_df.loc[len(tmp_df)] = [this_id, i]
        tmp_all_id.append(tmp_df)
    # print(tmp_all_num)

    if len(tmp_all_id) == 0:
        tmp_new_trackers_2 = tmp_new_trackers
    elif len(tmp_all_id) >= 1:
        critical_merge_df = pd.concat(tmp_all_id, ignore_index=True)
        critical_merge_df.split_frame += 1
        tmp_new_trackers_2 = {}
        ini_indx = 0
        for i in range(len(tmp_new_trackers)):
            if i not in critical_merge_df.tracker_id.values:
                tmp_new_trackers_2[ini_indx] = tmp_new_trackers[(i)]
                ini_indx += 1
            else:
                split_result = split_trackers(
                    i, critical_merge_df, tmp_new_trackers)
                for j in split_result:
                    tmp_new_trackers_2[ini_indx] = j
                    ini_indx += 1
    # new_summarize_2 = generate_summarize(new_trackers_2)

    tmp_new_summarize2 = generate_merge_summarize(tmp_new_trackers_2, dfs)

    return tmp_new_trackers_2, tmp_new_summarize2


def find_worms(ini_indx, summarize, banned_list=np.array([])):
    worms = []
    for i in range(len(ini_indx)):
        tmp_worm, banned_list = simple_find_nex(
            ini_indx[i], summarize, banned_list)
        if summarize.loc[ini_indx[i]].start_frame > 300:
            tmp_worm_prev, banned_list = simple_find_prev(
                ini_indx[i], summarize, banned_list)
            tmp_worm = np.concatenate((tmp_worm_prev[:-1], tmp_worm))
        worms.append(tmp_worm)
    return worms


def diagnosis_worms(worms, summarize, centroid, dfs, start_frame=300):
    """
    Diagnosis the worms
    """
    new_worms = []
    for i in range(len(worms)):
        tmp_worm = np.array([k for k in worms[i] if k != -1])
        if summarize.loc[tmp_worm[0]].start_frame > start_frame:
            print("worm %d start frame > %d" % (i, start_frame))
            continue
        if summarize.loc[tmp_worm[-1]].end_frame < np.max(dfs.index):
            if np.linalg.norm(summarize.loc[tmp_worm[-1]].end_centroid - centroid) < 850:
                print("worm %d end frame < max frame" % i)
                continue

        new_worms.append(tmp_worm)
    return new_worms


def diagnosis_worms2(worms, summarize, centroid, dfs, start_frame=300):
    """
    Diagnosis the worms
    """
    new_worms = []
    for i in range(len(worms)):
        tmp_worm = np.array([k for k in worms[i] if k != -1])
        if summarize.loc[tmp_worm[0]].start_frame > start_frame:
            print("worm %d start frame > %d" % (i, start_frame))
            continue
        tmp_summarize = summarize.loc[tmp_worm]
        tmp_summarize["end_distance"] = np.linalg.norm(
            np.array(tmp_summarize['end_centroid'].tolist()) - centroid, axis=1)
        end_tracker_df = tmp_summarize.loc[(tmp_summarize.end_distance > 900) & (
            tmp_summarize.right_type == "disappear")].sort_values(by='end_frame', ascending=True)
        if len(end_tracker_df) == 0:
            if summarize.loc[tmp_worm[-1]].end_frame == np.max(dfs.index):
                new_worms.append(tmp_worm)
            else:
                print("worm %d end frame < max frame" % i)
                continue
        else:
            end_tracker = end_tracker_df.iloc[0].tracker_id
            new_worms.append(
                tmp_worm[:(np.where(tmp_worm == end_tracker)[0][0]+1)])

    return new_worms


def diagnosis_worms_square(worms, summarize, centroid, dfs, radius, start_frame=300):
    """
    Diagnosis the worms
    """
    new_worms = []
    for i in range(len(worms)):
        tmp_worm = np.array([k for k in worms[i] if k != -1])
        if summarize.loc[tmp_worm[0]].start_frame > start_frame:
            print("worm %d start frame > %d" % (i, start_frame))
            continue
        tmp_summarize = summarize.loc[tmp_worm]
        # tmp_summarize['end_distance'] = np.max(np.abs(np.array(tmp_summarize['end_centroid'].tolist()) - centroid, axis=1))
        tmp_summarize['end_distance'] = (np.max(
            np.abs(np.array(tmp_summarize['end_centroid'].tolist()) - centroid), axis=1))
        # tmp_summarize["end_distance"] = np.linalg.norm(np.array(tmp_summarize['end_centroid'].tolist()) - centroid, axis=1)
        end_tracker_df = tmp_summarize.loc[(tmp_summarize.end_distance >= radius) & (
            tmp_summarize.right_type == "disappear")].sort_values(by='end_frame', ascending=True)
        if len(end_tracker_df) == 0:
            if summarize.loc[tmp_worm[-1]].end_frame == np.max(dfs.index):
                new_worms.append(tmp_worm)
            else:
                print("worm %d end frame < max frame" % i)
                continue
        else:
            end_tracker = end_tracker_df.iloc[0].tracker_id
            new_worms.append(
                tmp_worm[:(np.where(tmp_worm == end_tracker)[0][0]+1)])

    return new_worms


def worm2df(worm, trackers):
    dfs = []
    for i in worm:
        tmp_df = pd.DataFrame(trackers[i]['centroids'], columns=["x", "y"])
        tmp_frames = pd.Series(
            range(trackers[i]['start_frame'], trackers[i]['end_frame']+1), name="frames")
        tmp_df = pd.concat([tmp_frames, tmp_df], axis=1)
        tmp_trackers_id = pd.Series([i]*len(tmp_df), name="trackers_id")
        tmp_df = pd.concat([tmp_trackers_id, tmp_df], axis=1)
        tmp_ovals = pd.DataFrame(
            trackers[i]['ovals'], columns=["width", "height"])
        tmp_df = pd.concat([tmp_df, tmp_ovals], axis=1)
        tmp_df.set_index("frames", inplace=True)
        dfs.append(tmp_df)
    return pd.concat(dfs, axis=0)


def plot_worm(worm, centroid, trackers, radius, shape):
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, j in enumerate(worm):
        ax.plot(*np.array(trackers[j]['centroids']).T, label=j)
        if i < len(worm) - 1:
            last_centroid = trackers[j]['centroids'][-1]
            next_centroid = trackers[worm[i+1]]['centroids'][0]
            ax.plot([last_centroid[0], next_centroid[0]], [
                    last_centroid[1], next_centroid[1]], 'k--')

    ax.set_xlim(0, 3072)
    ax.set_ylim(0, 2048)

    ax.legend()
    plotly_fig = mpl_to_plotly(fig)
    if shape == "square":
        plotly_fig.add_shape(type="rect",
                             xref="x", yref="y",
                             x0=centroid[0]-radius, y0=centroid[1]-radius,
                             x1=centroid[0]+radius, y1=centroid[1]+radius,
                             line_color="black",)
    else:
        plotly_fig.add_shape(type="circle",
                            xref="x", yref="y",
                            x0=centroid[0]-radius, y0=centroid[1]-radius,
                            x1=centroid[0]+radius, y1=centroid[1]+radius,
                            line_color="black",)
    return plotly_fig


def write_results(subj, worms, trackers, centroid, radius, shape="square" ,p2final="./final_results"):
    os.path.isdir(p2final) or os.mkdir(p2final)
    p2subj = os.path.join(p2final, subj.lower())
    os.path.isdir(p2subj) and shutil.rmtree(p2subj)
    os.path.isdir(p2subj) or os.mkdir(p2subj)
    p2imgs = os.path.join(p2subj, "imgs")
    os.path.isdir(p2imgs) or os.mkdir(p2imgs)
    p2csvs = os.path.join(p2subj, "csvs")
    os.path.isdir(p2csvs) or os.mkdir(p2csvs)

    for i, j in enumerate(worms):
        tmp_fig = plot_worm(j, centroid, trackers, radius, shape=shape)
        # tmp_fig.write_image(os.path.join(p2imgs, "worms_%d.png" % (i)))
        pio.write_html(tmp_fig, file=os.path.join(
            p2imgs, "worms_%d.html" % (i)), auto_open=True)
        tmp_df = worm2df(j, trackers)
        tmp_df.to_csv(os.path.join(p2csvs, "worms_%d.csv" % (i)))
