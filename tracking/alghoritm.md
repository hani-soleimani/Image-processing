# Bead detection and tracking algorithm

## 1. Read the video and define the ROI

Each frame is read from the microscope video. The same fixed region of
interest is used for every frame:

```text
x = 74, y = 253, width = 279, height = 178
```

## 2. Convert the ROI to grayscale

The color ROI is converted from BGR to one grayscale channel:

```python
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
```

The grayscale image is passed directly to OpenCV's `SimpleBlobDetector`.
The final workflow does not create a custom binary threshold image.

## 3. Detect blob candidates

`SimpleBlobDetector` searches the grayscale ROI using its internal threshold
sweep. Candidate keypoints are estimated bright blob centers with a detected
size (`keypoint.size`). Candidates between **2 and 21 pixels**, inclusive, are
shown in the first-frame confirmation window.

## 4. Confirm targets in the first frame

The first frame displays numbered candidates. The user can:

- left-click real bead centers,
- right-click to remove the last selection,
- press Enter to accept the selected beads,
- press Escape to cancel.

Only confirmed beads become tracking targets. Unconfirmed detections do not
receive tracking IDs.

## 5. Match targets frame by frame

For each later frame, grayscale blob detection runs again. Each confirmed
track is matched to a candidate using nearest-neighbor distance. A match is
accepted only when the distance is strictly less than **25 pixels**.

Matches are one-to-one: a candidate and a track can each be used only once.
Matched detections keep their persistent track IDs. A target may be absent for
up to **5 consecutive frames**. If it is not recovered within that limit, its
track is removed. No motion prediction is used.

## 6. Save results

The output video draws the fixed ROI, detected bead circles, persistent IDs,
and the frame number. The CSV stores:

```text
frame_number, track_id, x, y, size
```

Coordinates refer to the full video frame, and `size` is the keypoint diameter
reported by `SimpleBlobDetector` in pixels.

## Methods intentionally not used

The final workflow does not use custom bright-spot thresholding, connected
components, circularity filtering, Kalman filtering, or CSRT tracking. A
separate CSRT experiment exists for single-target testing only.
