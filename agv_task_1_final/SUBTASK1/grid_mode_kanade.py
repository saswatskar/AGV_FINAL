"""
AGV Software Task - Subtask 1: Optical Flow
============================================
Two modes (press TAB to switch while running):

  MODE A  — Shi-Tomasi corners  (original)
  MODE B  — Fixed grid points   (uniform vector field)

Controls
--------
  q      quit
  TAB    toggle corner-detection ↔ grid mode
  r      reset / re-seed points
  d      toggle dense Farneback side-by-side (BONUS)
  +/-    (grid mode) increase / decrease grid spacing
"""

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
#  Bilinear patch sampler
# ══════════════════════════════════════════════════════════════

def _bilinear_patch(img: np.ndarray, cx: float, cy: float, half: int):
    H, W = img.shape
    xs = np.arange(-half, half + 1, dtype=np.float32) + cx
    ys = np.arange(-half, half + 1, dtype=np.float32) + cy
    if xs[0] < 0 or ys[0] < 0 or xs[-1] >= W - 1 or ys[-1] >= H - 1:
        return None
    x0 = np.floor(xs).astype(np.int32);  x1 = x0 + 1
    y0 = np.floor(ys).astype(np.int32);  y1 = y0 + 1
    ax = (xs - x0).astype(np.float32)
    ay = (ys - y0).astype(np.float32)
    Ia = img[np.ix_(y0, x0)].astype(np.float32)
    Ib = img[np.ix_(y1, x0)].astype(np.float32)
    Ic = img[np.ix_(y0, x1)].astype(np.float32)
    Id = img[np.ix_(y1, x1)].astype(np.float32)
    return (np.outer(1-ay, 1-ax)*Ia + np.outer(ay, 1-ax)*Ib +
            np.outer(1-ay,   ax)*Ic + np.outer(ay,   ax)*Id)


# ══════════════════════════════════════════════════════════════
#  Spatial gradients
# ══════════════════════════════════════════════════════════════

def _gradients(img: np.ndarray):
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32) / 8.0
    return (cv2.filter2D(img.astype(np.float32), -1, kx),
            cv2.filter2D(img.astype(np.float32), -1, kx.T))


# ══════════════════════════════════════════════════════════════
#  Lucas-Kanade tracker  (pyramidal, iterative)
# ══════════════════════════════════════════════════════════════

class LucasKanadeTracker:
    def __init__(self, win=21, levels=3, max_iter=20, eps=0.03, min_eig=1e-3):
        self.half     = win // 2
        self.levels   = levels
        self.max_iter = max_iter
        self.eps      = eps
        self.min_eig  = min_eig

    def _pyramid(self, img):
        pyr = [img.astype(np.float32)]
        for _ in range(self.levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def _lk_point(self, I1, Ix, Iy, I2, px, py):
        half = self.half
        p1   = _bilinear_patch(I1, px, py, half)
        if p1 is None: return 0.0, 0.0, False
        ix = _bilinear_patch(Ix, px, py, half)
        iy = _bilinear_patch(Iy, px, py, half)
        if ix is None or iy is None: return 0.0, 0.0, False

        AtA = np.array([[np.sum(ix*ix), np.sum(ix*iy)],
                        [np.sum(ix*iy), np.sum(iy*iy)]], dtype=np.float64)
        if np.linalg.eigvalsh(AtA)[0] < self.min_eig:
            return 0.0, 0.0, False
        AtA_inv = np.linalg.inv(AtA)

        vx = vy = 0.0
        for _ in range(self.max_iter):
            p2 = _bilinear_patch(I2, px+vx, py+vy, half)
            if p2 is None: return 0.0, 0.0, False
            It    = p2 - p1
            delta = AtA_inv @ np.array([-np.sum(ix*It), -np.sum(iy*It)])
            vx   += delta[0];  vy += delta[1]
            if delta[0]**2 + delta[1]**2 < self.eps**2:
                break
        return vx, vy, True

    def track(self, prev, curr, pts):
        """pts: (N,1,2) float32  →  next_pts, status  same shape"""
        pyr1 = self._pyramid(prev)
        pyr2 = self._pyramid(curr)
        N    = len(pts)
        disp = np.zeros((N, 2), np.float64)
        ok   = np.ones(N, bool)

        for lvl in range(self.levels, -1, -1):
            Ix, Iy = _gradients(pyr1[lvl])
            lscale = 2.0 ** lvl
            for i in range(N):
                if not ok[i]: continue
                px = pts.reshape(N,2)[i,0] / lscale + disp[i,0]
                py = pts.reshape(N,2)[i,1] / lscale + disp[i,1]
                dx, dy, valid = self._lk_point(pyr1[lvl], Ix, Iy, pyr2[lvl], px, py)
                if not valid: ok[i] = False
                else: disp[i] += [dx, dy]
            if lvl > 0: disp *= 2.0

        orig     = pts.reshape(N,2).astype(np.float64)
        next_pts = (orig + disp).astype(np.float32).reshape(N,1,2)
        return next_pts, ok.astype(np.uint8).reshape(N,1)

    def track_with_fb_check(self, prev, curr, pts, fb_thresh=1.5):
        """Forward-backward consistency filter to remove jittery tracks."""
        next_pts, status = self.track(prev, curr, pts)
        back_pts, back_status = self.track(curr, prev, next_pts)

        fb_err = np.linalg.norm(
            pts.reshape(-1,2) - back_pts.reshape(-1,2), axis=1)
        fb_ok  = (fb_err < fb_thresh).astype(np.uint8).reshape(-1,1)
        status = (status & fb_ok & back_status)
        return next_pts, status


# ══════════════════════════════════════════════════════════════
#  Point generators
# ══════════════════════════════════════════════════════════════

def shi_tomasi_points(gray, max_corners=150, quality=0.03, min_dist=14):
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_dist, blockSize=7)
    return pts   # (N,1,2) or None


def grid_points(h, w, spacing=40, margin=20):
    """
    Uniformly spaced grid of points across the frame.
    Returns (N,1,2) float32.
    """
    xs = np.arange(margin, w - margin, spacing)
    ys = np.arange(margin, h - margin, spacing)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return grid.reshape(-1, 1, 2)


# ══════════════════════════════════════════════════════════════
#  Drawing
# ══════════════════════════════════════════════════════════════

def draw_corner_tracks(frame, mask, old_pts, new_pts, status):
    """Classic coloured trails for corner mode."""
    good_new = new_pts[status.flatten()==1].reshape(-1,2)
    good_old = old_pts[status.flatten()==1].reshape(-1,2)
    for (x1,y1),(x0,y0) in zip(good_new.astype(int), good_old.astype(int)):
        ang    = np.arctan2(float(y1-y0), float(x1-x0))
        hue    = int(((ang + np.pi) / (2*np.pi)) * 179)
        col    = cv2.cvtColor(np.uint8([[[hue,230,220]]]),
                              cv2.COLOR_HSV2BGR)[0,0].tolist()
        mask  = cv2.line(mask,   (x1,y1),(x0,y0), col, 2)
        frame = cv2.circle(frame,(x1,y1), 3,       col, -1)
    return cv2.add(frame, mask), mask


def draw_grid_vectors(frame, old_pts, new_pts, status,
                      spacing, scale=3.0):
    """
    Draw each flow vector as an arrow from its grid origin.
    Colour encodes direction (HSV hue), brightness encodes magnitude.
    A semi-transparent overlay keeps the background visible.

    'scale' amplifies the arrow length so small motions are visible.
    """
    overlay = frame.copy()
    valid   = status.flatten() == 1
    old_v   = old_pts.reshape(-1,2)[valid]
    new_v   = new_pts.reshape(-1,2)[valid]

    for (ox,oy),(nx,ny) in zip(old_v.astype(float), new_v.astype(float)):
        dx, dy = nx - ox, ny - oy
        mag    = np.sqrt(dx*dx + dy*dy)
        if mag < 0.3:                       # dead-zone — no movement
            # draw a small static dot
            cv2.circle(overlay, (int(ox),int(oy)), 2, (60,60,60), -1)
            continue

        # colour: hue = direction, value = magnitude (capped)
        ang  = np.arctan2(dy, dx)
        hue  = int(((ang + np.pi) / (2*np.pi)) * 179)
        val  = int(min(255, mag * 20))
        col  = cv2.cvtColor(np.uint8([[[hue, 220, val]]]),
                            cv2.COLOR_HSV2BGR)[0,0].tolist()

        # arrow tip scaled by displacement magnitude
        tip_x = int(ox + dx * scale)
        tip_y = int(oy + dy * scale)

        # clamp to frame
        H, W  = frame.shape[:2]
        tip_x = max(0, min(W-1, tip_x))
        tip_y = max(0, min(H-1, tip_y))

        cv2.arrowedLine(overlay, (int(ox),int(oy)), (tip_x,tip_y),
                        col, 1, tipLength=0.35)

    # blend so background stays visible
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


# ══════════════════════════════════════════════════════════════
#  Dense flow bonus
# ══════════════════════════════════════════════════════════════

def dense_flow_vis(prev_gray, curr_gray):
    flow     = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv      = np.zeros((*curr_gray.shape,3), dtype=np.uint8)
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def run(video_path: str, show_dense: bool = False, save: bool = True):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"
    ret, frame = cap.read()
    assert ret

    h, w      = frame.shape[:2]
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker   = LucasKanadeTracker(win=21, levels=3, max_iter=20, eps=0.03)

    # ── state ──────────────────────────────────────────────
    CORNER_MODE = 0
    GRID_MODE   = 1
    mode        = GRID_MODE       # start in grid mode
    grid_spacing = 40
    dense_on    = show_dense

    prev_pts  = grid_points(h, w, grid_spacing)
    mask      = np.zeros_like(frame)           # trail layer (corner mode only)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_w  = w * 2 if dense_on else w
        writer = cv2.VideoWriter("output_flow.avi", fourcc, 25.0, (out_w, h))

    idx = 0
    mode_names = {CORNER_MODE: "Shi-Tomasi corners", GRID_MODE: "Fixed grid vectors"}
    print("TAB=switch mode  q=quit  r=reset  d=dense  +/-=grid spacing")

    while True:
        ret, frame = cap.read()
        if not ret: break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── track ───────────────────────────────────────────
        if prev_pts is not None and len(prev_pts) >= 2:
            next_pts, status = tracker.track_with_fb_check(
                prev_gray, curr_gray, prev_pts)
        else:
            next_pts = prev_pts
            status   = np.zeros((len(prev_pts) if prev_pts is not None else 0, 1),
                                dtype=np.uint8)

        # ── render ──────────────────────────────────────────
        vis = frame.copy()

        if mode == CORNER_MODE:
            vis, mask = draw_corner_tracks(vis, mask, prev_pts, next_pts, status)
            # keep only surviving points; re-detect if too few
            prev_pts = next_pts[status.flatten()==1].reshape(-1,1,2)
            if prev_pts is None or len(prev_pts) < 25:
                prev_pts = shi_tomasi_points(curr_gray)
                mask     = np.zeros_like(frame)

        else:  # GRID_MODE
            draw_grid_vectors(vis, prev_pts, next_pts, status, grid_spacing)
            # always reset grid every frame — fixed origins, fresh flow
            prev_pts = grid_points(h, w, grid_spacing)

        # ── dense bonus ─────────────────────────────────────
        if dense_on:
            dv   = dense_flow_vis(prev_gray, curr_gray)
            shown = np.hstack([vis, dv])
            cv2.putText(shown, "Dense Farneback (bonus)",
                        (w+8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            shown = vis

        # ── HUD ─────────────────────────────────────────────
        n = len(prev_pts) if prev_pts is not None else 0
        cv2.putText(shown,
                    f"{mode_names[mode]}  pts:{n}  frame:{idx}  "
                    f"{'[d=dense ON]' if dense_on else ''}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(shown, "TAB=mode  r=reset  d=dense  +/-=spacing  q=quit",
                    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        cv2.imshow("Optical Flow", shown)
        if writer: writer.write(shown)

        # ── keys ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 9:   # TAB
            mode     = 1 - mode
            mask     = np.zeros_like(frame)
            if mode == CORNER_MODE:
                prev_pts = shi_tomasi_points(curr_gray)
            else:
                prev_pts = grid_points(h, w, grid_spacing)
            print(f"Switched to: {mode_names[mode]}")
        elif key == ord('r'):
            mask = np.zeros_like(frame)
            if mode == CORNER_MODE:
                prev_pts = shi_tomasi_points(curr_gray)
            else:
                prev_pts = grid_points(h, w, grid_spacing)
        elif key == ord('d'):
            dense_on = not dense_on
            mask     = np.zeros_like(frame)
        elif key == ord('+') or key == ord('='):
            grid_spacing = min(80, grid_spacing + 5)
            prev_pts = grid_points(h, w, grid_spacing)
            print(f"Grid spacing: {grid_spacing}px")
        elif key == ord('-'):
            grid_spacing = max(15, grid_spacing - 5)
            prev_pts = grid_points(h, w, grid_spacing)
            print(f"Grid spacing: {grid_spacing}px")

        prev_gray = curr_gray.copy()
        idx += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"Done — {idx} frames processed.")


if __name__ == "__main__":
    import sys, os
    video = sys.argv[1] if len(sys.argv) > 1 else "test5.mp4"
    if not os.path.exists(video):
        print(f"Video not found: {video}"); sys.exit(1)
    run(video)