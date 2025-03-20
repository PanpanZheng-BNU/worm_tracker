import sys
import numpy as np
import multiprocessing
import argparse
import datetime



sys.path.append('./')
from detector.detector import detect
from conncat_videos import find_all_videos



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--p2v", type=str, help="Path to the directory containing the videos")
    parse.add_argument("--p2s", type=str, help="Path to store the concatenated videos")
    parse.add_argument("--pool", type=int, help="Number of processes to run", default=4)
    parse.add_argument("--vis", type=bool, help="Visualize the video", default=False)
    parse.add_argument("--img", type=bool, help="Visualize the video", default=False)
    parse.add_argument("--date", type=str, help="Date of the video", default=str(datetime.datetime.now()))
    args = parse.parse_args()

    p2v = args.p2v
    videos_dict = find_all_videos(p2v)
    dicts_list = []
    for k, v in videos_dict.items():
        dicts_list.append({k: v})

    print(dicts_list)
    # if len(dicts_list) > multiprocessing.cpu_count():
        # p = multiprocessing.Pool(multiprocessing.cpu_count())
    # else:
        # p = multiprocessing.Pool(len(dicts_list))
    if len(dicts_list) > args.pool:
        p = multiprocessing.Pool(args.pool)
    else:
        p = multiprocessing.Pool(len(dicts_list))
    p.starmap(detect, [(i,j, k, l,m) for i,j,k,l,m in zip(dicts_list,
                                                      925 * np.ones(len(dicts_list), dtype=int), 
                                                      [args.vis for _ in range(len(dicts_list))], 
                                                      [args.img for _ in range(len(dicts_list))],
                                                      [args.date for _ in range(len(dicts_list))])])
    # detect(dicts_list[1], 925)

