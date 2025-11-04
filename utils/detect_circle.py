import cv2
import numpy as np


def detect_red_circles(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Create a copy for drawing detected circles
    output = img.copy()

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define the range for red color in HSV
    # Red color wraps around in HSV, so it typically has two ranges.
    # Lower red range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Upper red range
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0.0)

    # 4. Noise reduction (Gaussian Blur is often effective)
    # This helps to smooth out the edges and make circles more detectable.
    blurred_mask = cv2.GaussianBlur(
        red_mask, (9, 9), 2
    )  # Kernel size and sigmaX can be adjusted

    # 5. Apply Hough Circle Transform
    # Parameters for HoughCircles:
    #   - image: 8-bit, single-channel (grayscale) image. The blurred_mask is perfect.
    #   - method: Currently, only HOUGH_GRADIENT is supported.
    #   - dp: Inverse ratio of the accumulator resolution to the image resolution.
    #         1 means same resolution as image, 2 means half resolution.
    #   - minDist: Minimum distance between the centers of the detected circles.
    #              This prevents multiple circles from being detected for the same actual circle.
    #   - param1: Upper threshold for the internal Canny edge detector.
    #   - param2: Threshold for center detection. Smaller values mean more false positives.
    #   - minRadius: Minimum circle radius (in pixels).
    #   - maxRadius: Maximum circle radius (in pixels).
    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=1,  # Increase to decrease accumulator resolution
        minDist=50,  # Minimum distance between circle centers
        param1=50,  # Canny edge detector upper threshold
        param2=30,  # Accumulator threshold for circle centers
        minRadius=5,  # Minimum radius of detected circles
        maxRadius=200,  # Maximum radius of detected circles
    )

    # Ensure some circles were found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            # Draw the outer circle
            cv2.circle(output, center, radius, (0, 255, 0), 2)  # Green circle outline
            # Draw the center of the circle
            cv2.circle(output, center, 2, (0, 0, 255), 3)  # Red dot for center

    # Display the results
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Red Mask", red_mask)
    # cv2.imshow("Detected Red Circles", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f"{image_path.split('.')[0]}_detect_circle.jpg", output)
    return center


if __name__ == "__main__":
    # Example usage
    image_path = "./N2_test1.jpg"  # Replace with your image file path
    print(detect_red_circles(image_path))
