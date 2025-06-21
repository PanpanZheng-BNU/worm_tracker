import argparse

import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Correcting points in an image.")
parser.add_argument(
    "--video_path",
    type=str,
    default="/Volumes/MyPassport/new_data/Qing/20250526/N2_Correct_0526.avi",
    help="Path to the video file.",
)
args = parser.parse_args()


# img = np.zeros((512, 512, 3), np.uint8)  # Create a blank image


cap = cv2.VideoCapture(args.video_path)
ret, img_orig = cap.read()

# img = img_orig.copy()  # Create a copy of the original image
global df
df = pd.DataFrame(columns=["x", "y"])

# img = cv2.imread(
#     "/Volumes/MyPassport/new_data/Qing/20250526/N2_Correct_0526.avi"
# )  # Load an image from file


print("Click on the image to select points. Press 'q' to quit.")


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for a left mouse button down event
        print(f"Clicked at coordinates: ({x}, {y})")
        df.loc[len(df)] = [x, y]
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the click
        cv2.imshow("image", img)  # Update the displayed image


for i in range(3):
    img = img_orig.copy()  # Reset the image to the original for each iteration
    cv2.namedWindow("image")  # Create a named window
    cv2.imshow("image", img)  # Display the image in the window
    cv2.setMouseCallback("image", draw_circle)

    while True:
        if cv2.waitKey(20) & (0xFF == ord("q")):
            break
        if len(df) % 6 == 0 and len(df) > i * 6:
            print("6 points selected, exiting...")
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows
    # cv2.destroyWindow("image")  # Close the current window

cm2pix = np.linalg.norm(
    df.iloc[::2].to_numpy() - df.iloc[1::2].to_numpy(), axis=1
).mean()
print(f"cm2pix: {cm2pix:.2f}")
