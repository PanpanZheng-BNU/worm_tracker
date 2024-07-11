import cv2, os

videoName = "./Naive2.mp4"
f2pics = "./pics"

os.path.isdir(f2pics) or os.mkdir(f2pics)
cap = cv2.VideoCapture(videoName)
frameNo = 0

for i in range(10):
    ret, frame = cap.read()
    if ret:
        name = os.path.basename(videoName).split('.')[0] + "_{}.jpg".format(frameNo)
        print("new frame capture ..." + name)

        cv2.imwrite(os.path.join(f2pics, name), frame)
        frameNo += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()


