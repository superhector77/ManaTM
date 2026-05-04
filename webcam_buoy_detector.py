"""
Real-Time Red Buoy Detector – Webcam
============================================
Opens your default webcam and highlights any red buoy-shaped
objects it finds in each frame.

Dependencies:
    pip install opencv-python numpy
"""

import cv2
import numpy as np


# ── Tunable parameters ────────────────────────────────────────────────────────

# HSV colour thresholds for red (red wraps around 0°/180°, so two ranges)
LOWER_RED1 = np.array([0,   120,  70])
UPPER_RED1 = np.array([10,  255, 255])
LOWER_RED2 = np.array([170, 120,  70])
UPPER_RED2 = np.array([180, 255, 255])

MIN_AREA        = 500    # px² – blobs smaller than this are ignored
MIN_CIRCULARITY = 0.55   # 1.0 = perfect circle; buoys are roughly circular
BLUR_KERNEL     = 5      # Gaussian pre-blur kernel size (must be odd)

# ─────────────────────────────────────────────────────────────────────────────


def build_red_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """Return a binary mask where red pixels are white."""
    mask = (
        cv2.inRange(hsv_frame, LOWER_RED1, UPPER_RED1) |
        cv2.inRange(hsv_frame, LOWER_RED2, UPPER_RED2)
    )
    # Morphological clean-up to remove small speckles and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_buoys(mask: np.ndarray):
    """
    Return a list of (cx, cy, radius) tuples for every buoy-like blob
    found in the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    buoys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        buoys.append((int(cx), int(cy), radius))

    return buoys


def annotate_frame(frame: np.ndarray, buoys: list) -> np.ndarray:
    """Draw detection circles and labels onto the frame."""
    for i, (cx, cy, radius) in enumerate(buoys):
        r = int(radius)
        # Outer ring
        cv2.circle(frame, (cx, cy), r,     (0, 0, 255), 3)
        # Centre dot
        cv2.circle(frame, (cx, cy), 5,     (0, 255,  0), -1)
        # Label
        label = f"Buoy #{i+1}  r={r}px"
        cv2.putText(frame, label,
                    (cx - r, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # HUD
    count_text = f"Buoys detected: {len(buoys)}"
    colour = (0, 255, 0) if len(buoys) == 0 else (0, 0, 255)
    cv2.putText(frame, count_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.putText(frame, "Press 'q' to quit | 'm' mask view",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    cap = cv2.VideoCapture(0)          # 0 = default laptop webcam
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. "
            "Check that no other app is using it, then try again.")

    print("Webcam opened.  Looking for red buoys …")
    print("  'q'  → quit")
    print("  'm'  → toggle binary mask view")

    show_mask = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed – exiting.")
            break

        # ── Detection pipeline ────────────────────────────
        blurred = cv2.GaussianBlur(frame,
                                   (BLUR_KERNEL, BLUR_KERNEL), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = build_red_mask(hsv)
        buoys   = find_buoys(mask)

        # ── Display ───────────────────────────────────────
        if show_mask:
            # Show the red-pixel mask side-by-side
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            annotated = annotate_frame(frame.copy(), buoys)
            display = np.hstack([annotated, mask_bgr])
        else:
            display = annotate_frame(frame.copy(), buoys)

        cv2.imshow("Red Buoy Detector", display)

        # ── Keyboard controls ─────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()
    print("Detector stopped.")


if __name__ == "__main__":
    main()
