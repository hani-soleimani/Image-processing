# Detection data

This folder contains the grayscale SimpleBlobDetector code and frame-1
inspection artifacts used to choose the bead size range.

- `bead_detection.py`: detector configuration and grayscale ROI utilities.
- `frame1_original.png`: source frame 1 from `tracking1.mp4`.
- `frame1_grayscale_roi.png`: exact BGR-to-grayscale ROI passed to the detector.
- `frame1_detections_10_21.png`: frame 1 detections retained by the inclusive
  10–21 pixel keypoint-size range.
- `frame1_detections.csv`: raw detector candidates and their sizes before the
  size comparison.

The detector receives a single-channel grayscale ROI. No tracking is performed
in this folder. The current fixed ROI is `(74, 253, 279, 178)`.
