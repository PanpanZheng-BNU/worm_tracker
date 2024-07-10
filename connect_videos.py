#!/Users/zhengpanpan/miniconda3/envs/cv/bin/python

import os
import cv2

# def findallvideos(path2data):
#     subject_file_dict = {}
#     for root, dirs, files in os.walk(path2data):
#         if len(files) > 0:
#             fil_fn = [fn for fn in files if fn[0] != "."]
#             subject_num = list(set([fn.split(".")[0].split("_")[1] for fn in fil_fn]))
#             for sn in subject_num:
#                 for fn in fil_fn:
#                     if sn in fn:
#                         if sn not in subject_file_dict:
#                             subject_file_dict[sn] = []
#                         subject_file_dict[sn].append(os.path.join(root, fn))
#     return subject_file_dict


def connect_videos(path2store,*args):
    p2s = path2store
    if not os.path.exists(p2s):
        os.makedirs(p2s)
    s_name = args[0] + ".mp4"
    cap = cv2.VideoCapture(args[1][0])
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(p2s, s_name), fourcc, 10, size)

    for v in args[1]:
        curr_v = cv2.VideoCapture(v)
        while curr_v.isOpened():
            # Get return value and curr frame of curr video
            r, frame = curr_v.read()
            if not r:
                break
                # Write the frame
            out.write(frame)
    out.release()

    file_name = args[0]
    # for j in args[1]:
    #     print(j)
    print(args[1])

if __name__ == "__main__":
    # a = findalltype("./Black_and_White")
    p2data = os.path.join("/Volumes/MyPassport","Data", "2024.3.25")

    subject_file_dict = {}
    for root, dirs, files in os.walk(p2data):
        if len(files) > 0:
            fil_fn = [fn for fn in files if fn[0] != "."]
            subject_num = list(set([fn.split(".")[0].split("_")[1] for fn in fil_fn]))
            for sn in subject_num:
                for fn in fil_fn:
                    if sn in fn:
                        if sn not in subject_file_dict:
                            subject_file_dict[sn] = []
                        subject_file_dict[sn].append(os.path.join(root, fn))
                        subject_file_dict[sn].sort()

    # print(subject_file_dict)
    for k in subject_file_dict.keys():
        s_name = k + ".mp4"
        cap = cv2.VideoCapture(subject_file_dict[k][0])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(os.path.join("./", s_name), fourcc, fps , size)

        for v in subject_file_dict[k]:
            curr_v = cv2.VideoCapture(v)
            while curr_v.isOpened():
                # Get return value and curr frame of curr video
                r, frame = curr_v.read()
                if not r:
                    break
                    # Write the frame
                out.write(frame)
        out.release()
