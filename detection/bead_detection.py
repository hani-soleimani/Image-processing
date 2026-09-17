"""Detect bright blobs directly from a grayscale ROI using SimpleBlobDetector."""

import argparse

import cv2

def load_image(image_path):
    """Load a BGR image and stop clearly if the file cannot be read."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def convert_to_grayscale(image):
    """Convert OpenCV's BGR image to a single grayscale channel."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def select_roi(image):
    """Return the selected (x, y, width, height), or None on cancellation."""
    window = "Select ROI - drag, then Enter or Space; C to cancel"
    print("Draw a rectangle, then press Enter or Space. Press C to cancel.")
    try:
        roi = tuple(map(int, cv2.selectROI(
            window, image, showCrosshair=True, fromCenter=False
        )))
    finally:
        cv2.destroyWindow(window)
    return roi if roi[2] > 0 and roi[3] > 0 else None


def create_detector():
    """Detect bright blobs; size is filtered separately after detection."""
    params = cv2.SimpleBlobDetector_Params()
    # Grayscale goes directly to detect(). OpenCV sweeps thresholds internally.
    params.minThreshold = 0
    params.maxThreshold = 256
    params.thresholdStep = 10
    params.minRepeatability = 2
    params.minDistBetweenBlobs = 5

    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = False


    params.filterByCircularity = False

    # Disable shape filters so only the final diameter limits are applied.
    params.filterByConvexity = False
    params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(params)


def detect_blobs(grayscale, roi):
    """Detect in the crop and translate keypoints to full-image coordinates."""
    x, y, width, height = roi
    if (x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > grayscale.shape[1]
            or y + height > grayscale.shape[0]):
        raise ValueError("ROI must be a nonempty rectangle inside the image.")
    crop = grayscale[y:y + height, x:x + width]
    keypoints = create_detector().detect(crop)
    # Keypoint size is the estimated blob diameter in pixels.
    return [cv2.KeyPoint(k.pt[0] + x, k.pt[1] + y, k.size) for k in keypoints]


def filter_by_size(keypoints, min_size=10, max_size=21):
    """Keep inclusive keypoint diameters and preserve IDs from detection order."""
    # SimpleBlobDetector has no direct minSize/maxSize parameters.
    # Filter its returned keypoint.size values without changing the detector.
    return [
        (number, point) for number, point in enumerate(keypoints, 1)
        if min_size <= point.size <= max_size
    ]

def draw_blobs(original, blobs, roi):
    """Draw green blob circles and a cyan ROI on a copy of the original."""
    annotated = cv2.drawKeypoints(
        original, [point for _, point in blobs], None, (0, 255, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    x, y, width, height = roi
    cv2.rectangle(
        annotated, (x, y), (x + width - 1, y + height - 1), (255, 255, 0), 2
    )
    # Show only blob numbers; the terminal table lists their sizes.
    # Preserve original IDs so the image and terminal table match.
    for number, point in blobs:
        label = str(number)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        text_x = max(0, min(round(point.pt[0] + point.size / 2),
                            annotated.shape[1] - text_width))
        text_y = min(annotated.shape[0] - 1,
                     max(text_height, round(point.pt[1] - point.size / 2)))
        cv2.putText(
            annotated, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to the microscope image")
    args = parser.parse_args()
    try:
        # Load and convert before selecting the region to analyze.
        original = load_image(args.image_path)
        grayscale = convert_to_grayscale(original)
        roi = select_roi(original)
        if roi is None:
            print("ROI selection cancelled. No detection performed.")
            return
        keypoints = detect_blobs(grayscale, roi)
    except ValueError as error:
        parser.exit(1, f"Error: {error}\n")

    # Filter by diameter, keeping the detector's original numbering.
    min_size = 10
    max_size = 21
    blobs = filter_by_size(keypoints, min_size, max_size)

    print(f"ROI (x, y, width, height): {roi}")
    print(f"SimpleBlobDetector detections: {len(keypoints)}")
    print(f"After size filter {min_size}-{max_size} pixels: {len(blobs)}")
    print("Blob ID | Size (pixels)")
    for number, point in blobs:
        print(f"{number:7d} | {point.size:.2f}")

    annotated = draw_blobs(original, blobs, roi)
    window = "Detected beads - size 10-21 pixels"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, annotated)
        cv2.resizeWindow(window, original.shape[1], original.shape[0])
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






