# Image Processing

A small OpenCV project for detecting beads in microscope video and tracking the
beads confirmed by the user in the first frame.

## Project layout

```text
detection/
  bead_detection.py          Grayscale SimpleBlobDetector configuration
tracking/
  bead_tracking_manual_help.py  First-frame confirmation and tracking
requirements.txt
README.md
```

## Setup

```powershell
cd "D:\hani\Image Processing\image-processing"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use a desktop Python environment because the program opens OpenCV windows.

## Run

```powershell
.\.venv\Scripts\python.exe -m tracking.bead_tracking_manual_help `
  "D:\hani\Image Processing\Image detection\PIC\tracking1.mp4" `
  --output "results\tracking1_manual_help.mp4" `
  --csv "results\tracking1_manual_help.csv"
```

The program reads frame 1, converts the fixed ROI to grayscale, and detects
candidate blobs with the SimpleBlobDetector. Candidates are limited to sizes
2–21 pixels. The confirmation window uses orange candidate circles:

- Left-click adds a bead center.
- Right-click removes the last selection.
- Enter accepts the selected beads and starts tracking.
- Escape cancels without saving output.

Only confirmed beads receive IDs and are tracked. Matching uses nearest distance
under 25 pixels, retains tracks for up to five missed frames, and never creates
IDs for unconfirmed detections. No Kalman filter is used.

The annotated output shows the ROI, bead circles, track IDs, and frame number.
The CSV columns are `frame_number,track_id,x,y,size`.

The default ROI is `(74, 253, 279, 178)`. Update the constant in the tracker
when working with videos that use a different camera framing.

## Current limitations

This is an experimental tracker. Similar beads can exchange identities when
they cross, and a bead that moves more than 25 pixels between frames can be
missed. The first-frame confirmation is intentional: it lets the user reject
reflections and add real beads the detector did not propose.
