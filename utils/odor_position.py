import cv2,os
import numpy as np
import pandas as pd
import argparse
# img = np.zeros((512, 512, 3), np.uint8)  # Create a blank image

parser = argparse.ArgumentParser(description="Odor position measurement tool")
parser.add_argument(
    "--video_path",
    type=str,
    help="Path to the video file",
)
args = parser.parse_args()
# print(args.video_path)

cap = cv2.VideoCapture(args.video_path)
ret, img = cap.read()

df = pd.DataFrame(columns=["x", "y"])

# img = cv2.imread(
#     "/Volumes/MyPassport/new_data/Qing/20250526/N2_Correct_0526.avi"
# )  # Load an image from file
cv2.namedWindow("image")  # Create a named window
cv2.imshow("image", img)  # Display the image in the window


print("Click on the image to select points. Press 'q' to quit.")
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for a left mouse button down event
        print(f"Clicked at coordinates: ({x}, {y})")
        df.loc[len(df)] = [x, y]
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the click
        cv2.imshow("image", img)  # Update the displayed image


cv2.setMouseCallback("image", draw_circle)


while True:
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()  # Close all OpenCV windows

# cm2pix = np.linalg.norm(
    # df.iloc[::2].to_numpy() - df.iloc[1::2].to_numpy(), axis=1
# ).mean()
# print(f"cm2pix: {cm2pix:.2f}")
df.to_csv(args.video_path.split(os.sep)[-1].replace(".avi", ".csv"), index=False)

