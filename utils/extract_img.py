import argparse
import os

import cv2


def extract_image_from_video(video_path, frame_number):
    """
    Extracts a single frame from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_number (int): The frame number to extract.

    Returns:
        numpy.ndarray: The extracted frame as an image, or None if the frame could not be extracted.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    cv2.imwrite(
        os.path.join(
            # os.path.dirname(video_path),
            f"{video_path.split('/')[-1].split('.')[0]}.jpg",
        ),
        frame,
    )
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_number} from video {video_path}")
        return None

    return frame


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Extract a frame from a video file.")
    parser.add_argument("--p2v", type=str, help="Path to the video file.")
    parser.add_argument("--frame", type=int, help="frame number to extract", default=1)

    # parser.add_argument(
    #     "frame_number",
    #     type=int,
    #     help="The frame number to extract (0-indexed).",
    #     default=1,
    # )
    args = parser.parse_args()
    video_path = args.video_path
    # frame_number = args.frame_number
    frame = args.frame
    extract_image_from_video(video_path, frame)
