import cv2, multiprocessing, os
import pandas as pd
import tqdm


def find_all_videos(p2v):
    """
    find_all_videos(p2v)
    :param p2v: path to the directory containing the videos
    :return: all_file_dict: dictionary of all the videos to each subject
    """
    # Iterate through the directory and find all the videos
    all_dict = {}
    for root, dirs, files in os.walk(p2v):
        videos = []  # list of all the videos
        if len(files) > 0:  # if there are files in the directory
            videos.extend(
                [file for file in files if (".avi" in file) and file[0] != "."]
            )  # add the video to the list
        sub_id = [
            video.split(".")[0] for video in videos if len(video.split("_")) <= 4
        ]  # get the subject id
        tmp_dict = {
            id: sorted([os.path.join(root, v) for v in videos if id in v])
            for id in sub_id
        }  # create a dictionary of all the videos to each subject
        all_dict.update(tmp_dict)  # update the dictionary

    dicts_list = []
    for k, v in all_dict.items():
        dicts_list.append({k: v})
    return dicts_list


def concat_videos(p2vs, p2s):
    """
    concat_videos(p2v): Concatenate all the videos of a subject
    :param p2vs: path to all the videos
    :param p2s: path to store the concatenated videos
    :return: None
    """

    cap = cv2.VideoCapture(p2vs[0])
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    video = cv2.VideoWriter(p2s, fourcc, fps, size)
    for vi in tqdm.tqdm(range(len(p2vs))):
        curr_v = cv2.VideoCapture(p2vs[vi])
        while curr_v.isOpened():
            r, frame = curr_v.read()
            if not r:
                break
            video.write(frame)
    video.release()
    print(f"Video saved to {p2s}")


if __name__ == "__main__":
    # p2v = "/Volumes/MyPassport/new_data/2025.1.16"
    p2v = "/Volumes/MyPassport/new_data/2025.1.16"

    os.path.isdir("./concat_video") or os.makedirs("./concat_video")
    # all_dict = find_all_videos(p2v)
    print(find_all_videos(p2v))
    # args = [(v, os.path.join("./concat_video",k +".avi")) for k, v in all_dict.items()]
    # pool = multiprocessing.Pool()
    # pool.starmap(concat_videos, args)
    # pool.close()
