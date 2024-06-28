import os, cv2
def findalltype(dir):
    all_files = os.listdir(dir)
    all_files.sort()
    all_type = []
    files_group_by = []

    for file in all_files:
        exclude_extension = file.split(".")[0]
        if exclude_extension.split("_")[1] not in all_type:
            all_type.append(exclude_extension.split("_")[1])
            files_group_by.append([])
            files_group_by[-1].append(os.path.join(dir, file))
        else:
            files_group_by[-1].append(os.path.join(dir, file))

    return all_type, files_group_by

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
    a = findalltype("./Black_and_White")
    print(a)
    # connect_videos("./p2con",a[0][0], a[1][0])
