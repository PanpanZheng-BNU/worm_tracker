#!/Users/zhengpanpan/miniconda3/envs/worm-tracker/bin/python
import argparse
import multiprocessing
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append("./")
from detector.concat_videos import find_all_videos
from detector.detector import detect
from tracker.simple_tracking import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--p2vs", type=str, help="Path to the directory containing the videos"
    )
    parse.add_argument(
        "--p2det",
        type=str,
        help="Path to store the concatenated videos",
        default="./detect_results",
    )
    parse.add_argument("--pool", type=int, help="Number of processes", default=4)
    parse.add_argument(
        "--vis", type=str2bool, help="Visualize the video", default="false"
    )
    parse.add_argument(
        "--img",
        type=str2bool,
        help="Whether store the detect imgs at each frame",
        default=False,
    )
    parse.add_argument(
        "--video", type=str2bool, help="whether store the videos", default=False
    )
    parse.add_argument(
        "--date",
        type=str,
        help="Date of the video, to prevent name conflict and overwriting",
        default=datetime.strftime(datetime.now(), "%m.%d"),
    )
    parse.add_argument(
        "--radius",
        type=int,
        help="Radius of the ROI circle, default is 900",
        default=900,
    )

    parse.add_argument(
        "--p2trackers",
        type=str,
        help="Path to store the simple trackers",
        default="./simple_trackers_result",
    )
    return parse.parse_args()


def detect_main(args):
    dicts_list = find_all_videos(
        args.p2vs
    )  # obtain all paths to the videos in the specific directory
    # Set the size of parallel pool
    print(len(dicts_list))
    if len(dicts_list) > args.pool:
        p = multiprocessing.Pool(args.pool)
    else:
        p = multiprocessing.Pool(len(dicts_list))

    p.starmap(
        detect,
        [
            (vdict, p2s, radius, vis, img, video, date)
            for vdict, p2s, radius, vis, img, video, date in zip(
                dicts_list,
                [args.p2det for _ in range(len(dicts_list))],
                args.radius * np.ones(len(dicts_list), dtype=int),
                [args.vis for _ in range(len(dicts_list))],
                [args.img for _ in range(len(dicts_list))],
                [args.video for _ in range(len(dicts_list))],
                [args.date for _ in range(len(dicts_list))],
            )
        ],
    )


if __name__ == "__main__":
    # print(find_all_videos("/Volumes/MyPassport/new_data/2025.3.19")[1])
    args = create_parse()
    print("=" * 10 + "start detect" + "=" * 10)
    detect_main(args)
    print("=" * 10 + "start simple tracking" + "=" * 10)
    os.path.isdir(args.p2trackers) or os.makedirs(args.p2trackers)
    mv_centroids(args)
    generate_trackers_and_long_df(args)

    # # detect(dicts_list[1], 925)
