import cv2
import numpy as np

# Load your image (replace 'your_image.png' with your image path)
image = cv2.imread("./N2_test2.jpg")
if image is None:
    print("Error: Could not load image.")
    exit()

output_image = image.copy()  # For drawing results

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(
    blurred_image, 50, 150
)  # Apply Canny on blurred image for better edge quality

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print(len(contours))
for contour in contours:
    # 1. Filter by Area: Remove very small or very large contours that are unlikely to be your object.
    area = cv2.contourArea(contour)
    if area < 500 or area > 50000:  # Adjust min/max area thresholds as needed
        continue

    # 2. Approximate Polygon: Simplify the contour to find vertices.
    # A rectangle has 4 vertices. A rounded rectangle's approximation might have more,
    # or even 4 if the rounding is very subtle.
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour, 0.04 * perimeter, True
    )  # Adjust epsilon (0.04*perimeter)

    # 3. Analyze the Approximated Polygon (first pass)
    # A true rectangle would have 4 vertices. A rounded one might have slightly more,
    # or still 4 if the epsilon for approxPolyDP is large.
    num_vertices = len(approx)

    # Filter contours that are roughly rectangular (e.g., 4 to 8 vertices)
    # if not (
    #     4 <= num_vertices <= 8
    # ):  # This range can be adjusted based on rounding severity
    #     continue

    # 4. Check Aspect Ratio and Bounding Box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    # if not (
    #     0.5 < aspect_ratio < 2.0
    # ):  # Adjust range for your expected rectangles (e.g., 1.0 for squares)
    #     continue

    # 5. More Advanced Check: Curvature analysis or fitting
    # A rounded rectangle has long straight segments and curved corners.
    # This is more complex and involves analyzing the contour's curvature.
    # One approach:
    #   a. Find corners using methods like Harris or by detecting high curvature points.
    #   b. Verify that segments between corners are relatively straight.
    #   c. Verify that regions around corners show consistent curvature (arcs).

    # Simplistic heuristic for roundedness:
    # Compare the contour's area to the area of its bounding box.
    # For a perfect rectangle, these would be very close. For a rounded rectangle,
    # the contour area might be slightly less than its bounding box area,
    # but not as much as, say, a circle or triangle.
    bbox_area = w * h
    solidity = float(area) / bbox_area
    # A very rough heuristic: solidity for a circle is ~0.785, square is 1.
    # A rounded rectangle will be between a square and a circle's solidity.
    # This range is highly dependent on how rounded the corners are.
    # if not (0.85 < solidity < 0.99):  # Adjust these bounds carefully
    #     continue

    # 6. Check for parallel lines (optional, but robust)
    # You could use Hough Transform to detect lines, then check for 4 lines and parallelism.
    # Or, if `approx` gives 4 vertices, verify the angles are close to 90 degrees.
    # If it gives more, try to identify the 4 main "corner" regions and the straight lines.

    # If all checks pass (you'll need to refine them), draw the detected shape
    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # Draw in green
    cv2.putText(
        output_image,
        "Rounded Rect",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

# Display the result
cv2.imshow("Detected Rounded Rectangles", output_image)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
