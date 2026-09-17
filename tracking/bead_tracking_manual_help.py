"""Interactive first-frame bead confirmation followed by automatic tracking."""

import argparse
import csv
import math
from pathlib import Path

import cv2

from detection.bead_detection import convert_to_grayscale, create_detector

ROI = (74, 253, 279, 178)
MAX_MATCH_DISTANCE = 25
MAX_MISSED_FRAMES = 5
MIN_DETECTION_SIZE = 2
MAX_DETECTION_SIZE = 21


def detect_candidates(frame):
    """Detect all grayscale candidates in the fixed ROI; no size cutoff yet."""
    x, y, width, height = ROI
    gray = convert_to_grayscale(frame)
    roi_gray = gray[y:y + height, x:x + width]
    points = create_detector().detect(roi_gray)
    return [cv2.KeyPoint(p.pt[0] + x, p.pt[1] + y, p.size) for p in points]


def choose_beads(frame, candidates):
    """Show numbered candidates and let the user confirm or add centers."""
    selected = []
    window = "Confirm beads - left click add, right click undo, Enter accept, Esc cancel"

    def redraw():
        view = frame.copy()
        x, y, width, height = ROI
        cv2.rectangle(view, (x, y), (x + width - 1, y + height - 1), (255, 255, 0), 2)
        for number, point in enumerate(candidates, 1):
            radius = max(3, round(point.size / 2))
            cv2.circle(view, (round(point.pt[0]), round(point.pt[1])), radius, (0, 180, 255), 1)
            cv2.putText(view, str(number), (round(point.pt[0]) + 3, round(point.pt[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        for number, (cx, cy) in enumerate(selected, 1):
            cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(view, f"T{number}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1)
        cv2.putText(view, f"Candidates: {len(candidates)} | Confirmed: {len(selected)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return view

    def mouse(event, mouse_x, mouse_y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((mouse_x, mouse_y))
        elif event == cv2.EVENT_RBUTTONDOWN and selected:
            selected.pop()

    print("Frame 1 help: click true bead centers; right-click undoes; Enter accepts; Esc cancels.")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            cv2.imshow(window, redraw())
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                return selected
            if key == 27:
                return None
    finally:
        cv2.destroyWindow(window)


def match_targets(previous, candidates):
    """Match only confirmed targets; unmatched detections never become new IDs."""
    pairs = []
    for bead_id, track in previous.items():
        for index, point in enumerate(candidates):
            distance = math.hypot(point.pt[0] - track["position"][0],
                                  point.pt[1] - track["position"][1])
            if distance < MAX_MATCH_DISTANCE:
                pairs.append((distance, bead_id, index))
    assigned, used_ids = {}, set()
    for _, bead_id, index in sorted(pairs):
        if bead_id not in used_ids and index not in assigned:
            assigned[index] = bead_id
            used_ids.add(bead_id)
    visible = [(assigned[index], point) for index, point in enumerate(candidates)
               if index in assigned]
    current = {bead_id: {"position": point.pt, "missed": 0}
               for bead_id, point in visible}
    for bead_id, track in previous.items():
        if bead_id not in current and track["missed"] < MAX_MISSED_FRAMES:
            current[bead_id] = {"position": track["position"],
                                 "missed": track["missed"] + 1}
    return visible, current


def annotate(frame, visible, frame_number):
    """Draw confirmed target circles, IDs, ROI, and frame number."""
    result = frame.copy()
    x, y, width, height = ROI
    cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (255, 255, 0), 2)
    for bead_id, point in visible:
        cv2.drawKeypoints(result, [point], result, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(result, str(bead_id), (round(point.pt[0] + point.size / 2),
                    round(point.pt[1] - point.size / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, f"Frame {frame_number} | Confirmed targets: {len(visible)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.csv).exists():
        parser.error("Output files already exist; choose new paths.")

    cap = cv2.VideoCapture(args.video_path)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read frame 1")
    candidates = detect_candidates(frame)
    selected = choose_beads(frame, candidates)
    if selected is None or not selected:
        cap.release()
        print("No beads confirmed; no tracking output saved.")
        return

    # Initialize confirmed IDs from the clicked positions, then use detections only afterward.
    previous = {i: {"position": position, "missed": 0}
                for i, position in enumerate(selected, 1)}
    fps = cap.get(cv2.CAP_PROP_FPS) or 18
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (frame.shape[1], frame.shape[0]))
    csv_file = open(args.csv, "x", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_number", "track_id", "x", "y", "size"])
    frame_number = 0
    try:
        while ok:
            frame_number += 1
            if frame_number == 1:
                visible = [(i, cv2.KeyPoint(x, y, 1)) for i, (x, y) in enumerate(selected, 1)]
            else:
                visible, previous = match_targets(previous, detect_candidates(frame))
            # Keep clicked frame-1 positions visible; later frames use detector keypoints.
            for track_id, point in visible:
                csv_writer.writerow([frame_number, track_id, point.pt[0], point.pt[1], point.size])
            result = annotate(frame, visible, frame_number)
            writer.write(result)
            if not args.no_display:
                cv2.imshow("Bead tracking - confirmed targets", result)
                if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF in (27, ord("q")):
                    break
            ok, frame = cap.read()
    finally:
        cap.release(); writer.release(); csv_file.close(); cv2.destroyAllWindows()
    print(f"Confirmed targets: {len(selected)}; processed frames: {frame_number}")
    print(f"Saved video: {args.output}")
    print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()




