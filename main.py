import sys


sys.path.append('./')
from detector.detector import detect
from conncat_videos import find_all_videos


if __name__ == "__main__":
    p2v = "/Volumes/MyPassport/new_data/2025.1.12"
    videos_dict = find_all_videos(p2v)
    dicts_list = []
    for k, v in videos_dict.items():
        dicts_list.append({k: v})

    print(dicts_list)
    detect(dicts_list[1], 925)

