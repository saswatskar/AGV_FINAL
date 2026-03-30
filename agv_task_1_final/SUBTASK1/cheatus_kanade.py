"""
AGV Software Task - Subtask 1: Optical Flow
============================================
Sparse  : cv2.calcOpticalFlowPyrLK  (Lucas-Kanade)
BONUS   : cv2.calcOpticalFlowFarneback  (Dense)

Controls while running
----------------------
q  - quit
r  - re-detect features
d  - toggle dense flow side-by-side
"""

import cv2
import numpy as np

# ── Feature detection params (Shi-Tomasi) ──────────────────────────────────
FEATURE_PARAMS = dict(
    maxCorners   = 200,
    qualityLevel = 0.01,
    minDistance  = 10,
    blockSize    = 7,
)

# ── Lucas-Kanade params ────────────────────────────────────────────────────
LK_PARAMS = dict(
    winSize  = (51, 51),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

REDETECT_BELOW = 20   # re-detect when fewer than this many points survive


# ── Dense flow visualisation (Farneback BONUS) ─────────────────────────────

def dense_flow_vis(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    flow     = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv      = np.zeros((*curr_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2          # hue  = direction
    hsv[..., 1] = 255                              # full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── Draw coloured trails ───────────────────────────────────────────────────

def draw_tracks(frame, mask, old_pts, new_pts, status):
    good_new = new_pts[status.flatten() == 1].reshape(-1, 2)
    good_old = old_pts[status.flatten() == 1].reshape(-1, 2)

    for (x1, y1), (x0, y0) in zip(good_new.astype(int), good_old.astype(int)):
        ang    = np.arctan2(float(y1 - y0), float(x1 - x0))
        hue    = int(((ang + np.pi) / (2 * np.pi)) * 179)
        colour = cv2.cvtColor(
            np.uint8([[[hue, 230, 220]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        mask  = cv2.line(mask,    (x1, y1), (x0, y0), colour, 2)
        frame = cv2.circle(frame, (x1, y1), 4,         colour, -1)

    return cv2.add(frame, mask), mask


# ── Main loop ──────────────────────────────────────────────────────────────

def run(video_path: str, show_dense: bool = True, save: bool = True):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video: {video_path}"

    ret, frame = cap.read()
    assert ret, "Cannot read first frame."

    h, w      = frame.shape[:2]
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts  = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    trail_mask = np.zeros_like(frame)
    dense_on   = show_dense

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_w  = w * 2 if dense_on else w
        writer = cv2.VideoWriter("output_flow.avi", fourcc, 25.0, (out_w, h))

    idx = 0
    print("Controls: q=quit  r=re-detect  d=toggle dense")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Sparse Lucas-Kanade (built-in) ──────────────────────────────
        if prev_pts is not None and len(prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **LK_PARAMS)

            # forward-backward consistency check for extra stability
            prev_pts_back, _, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, prev_gray, next_pts, None, **LK_PARAMS)
            fb_err = np.linalg.norm(
                prev_pts - prev_pts_back, axis=2).flatten()
            good   = (status.flatten() == 1) & (fb_err < 1.0)

            sparse_vis, trail_mask = draw_tracks(
                frame.copy(), trail_mask,
                prev_pts, next_pts,
                good.astype(np.uint8))

            prev_pts = next_pts[good].reshape(-1, 1, 2)
        else:
            sparse_vis = frame.copy()

        # re-detect if too few points remain
        if prev_pts is None or len(prev_pts) < REDETECT_BELOW:
            prev_pts   = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
            trail_mask = np.zeros_like(frame)

        # ── Dense Farneback (BONUS) ─────────────────────────────────────
        if dense_on:
            dv    = dense_flow_vis(prev_gray, curr_gray)
            shown = np.hstack([sparse_vis, dv])
            cv2.putText(shown, "BONUS: Dense Farneback",
                        (w + 8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
        else:
            shown = sparse_vis

        n = len(prev_pts) if prev_pts is not None else 0
        cv2.putText(shown, f"Sparse LK (PyrLK)  pts:{n}  frame:{idx}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        desired_width = 1500  # you can change this (1000–1500 works well)
        h, w = shown.shape[:2]
        scale = desired_width / w
        shown_big = cv2.resize(shown, None, fx=scale, fy=scale)

        cv2.imshow("Optical Flow", shown_big)

        if writer:
            writer.write(shown)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('r'):
            prev_pts   = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
            trail_mask = np.zeros_like(frame)
        elif key == ord('d'):
            dense_on   = not dense_on
            trail_mask = np.zeros_like(frame)

        prev_gray = curr_gray.copy()
        idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {idx} frames processed.")


if __name__ == "__main__":
    import sys, os
    video = sys.argv[1] if len(sys.argv) > 1 else "test5.mp4"
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        sys.exit(1)
    run(video)