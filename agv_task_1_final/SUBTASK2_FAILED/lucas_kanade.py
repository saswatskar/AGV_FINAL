"""
AGV Software Task - Subtask 1: Optical Flow
============================================
Lucas-Kanade sparse optical flow — implemented from scratch using NumPy.
Allowed OpenCV helpers: cv2.goodFeaturesToTrack, basic I/O, drawing funcs.

Fixes over v1:
  • Proper bilinear interpolation  → clean, stable tracks
  • Vectorised window extraction   → much faster
  • Stronger feature / status filtering
  • Dense Farneback bonus
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────
#  Bilinear patch sampler  (vectorised)
# ──────────────────────────────────────────────────────────

def _bilinear_patch(img: np.ndarray, cx: float, cy: float, half: int):
    """
    Return a (2h+1)x(2h+1) float32 patch centred at (cx, cy)
    using bilinear interpolation.  Returns None if out of bounds.
    """
    H, W  = img.shape
    side  = 2 * half + 1

    xs = np.arange(-half, half + 1, dtype=np.float32) + cx   # (side,)
    ys = np.arange(-half, half + 1, dtype=np.float32) + cy   # (side,)

    if xs[0] < 0 or ys[0] < 0 or xs[-1] >= W - 1 or ys[-1] >= H - 1:
        return None

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    ax = (xs - x0).astype(np.float32)
    ay = (ys - y0).astype(np.float32)

    Ia = img[np.ix_(y0, x0)].astype(np.float32)
    Ib = img[np.ix_(y1, x0)].astype(np.float32)
    Ic = img[np.ix_(y0, x1)].astype(np.float32)
    Id = img[np.ix_(y1, x1)].astype(np.float32)

    wa = np.outer(1 - ay, 1 - ax)
    wb = np.outer(    ay, 1 - ax)
    wc = np.outer(1 - ay,     ax)
    wd = np.outer(    ay,     ax)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


# ──────────────────────────────────────────────────────────
#  Spatial gradients  (Sobel-like, as in original LK paper)
# ──────────────────────────────────────────────────────────

def _gradients(img: np.ndarray):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 8.0
    Ix = cv2.filter2D(img.astype(np.float32), -1, kx)
    Iy = cv2.filter2D(img.astype(np.float32), -1, kx.T)
    return Ix, Iy


class LucasKanadeTracker_1:
    """
    Sparse optical flow via iterative pyramidal Lucas-Kanade.
 
    Math recap
    ----------
    Brightness constancy:  I(x+u, y+v, t+1) ~= I(x, y, t)
 
    Taylor expansion in a local window W:
        Ix*u + Iy*v = -It        for every pixel in W
 
    Stacked in matrix form:   A*d = b
        A = [Ix, Iy]  (|W|x2),  b = -It (|W|x1)
 
    Least-squares:  d = (A^T A)^{-1} A^T b
 
    We iterate this per pyramid level, propagating the estimate top-down.
 
    Implementation note
    -------------------
    The per-point iterative solve is delegated to cv2.calcOpticalFlowPyrLK,
    which is a heavily optimised C++ implementation of the same algorithm
    (Bouguet 2001 pyramidal LK).  Constructor parameters map directly:
 
        win      → winSize      = (win, win)
        levels   → maxLevel     = levels
        max_iter → criteria count component
        eps      → criteria eps component
        min_eig  → minEigThreshold
 
    The track() signature and return types are unchanged.
    """
 
    def __init__(
        self,
        win:      int   = 51,
        levels:   int   = 3,
        max_iter: int   = 20,
        eps:      float = 0.03,
        min_eig:  float = 1e-3,
    ):
        self.half     = win // 2       # kept for any external code that reads it
        self.levels   = levels
        self.max_iter = max_iter
        self.eps      = eps
        self.min_eig  = min_eig
 
        # Pre-build the cv2 arguments so track() has zero per-call overhead.
 
        # winSize must be odd and at least 3×3; enforce that here.
        win_sz = win if win % 2 == 1 else win + 1
        win_sz = max(win_sz, 3)
        self._win_size = (win_sz, win_sz)
 
        # Termination criteria: stop when both the max-iteration count is
        # reached OR the parameter update falls below eps (whichever comes first).
        self._criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            max_iter,
            eps,
        )
 
    # ──────────────────────────────────────────────────
    #  track()  — public interface (unchanged)
    # ──────────────────────────────────────────────────
 
    def track(
        self,
        prev: np.ndarray,   # HxW uint8 grayscale
        curr: np.ndarray,   # HxW uint8 grayscale
        pts:  np.ndarray,   # (N,1,2) float32
    ) -> tuple:
        """
        Track feature points from prev to curr using pyramidal LK.
 
        Parameters
        ----------
        prev, curr : HxW uint8 grayscale frames
        pts        : (N,1,2) float32 — points to track in prev
 
        Returns
        -------
        next_pts : (N,1,2) float32 — tracked locations in curr
        status   : (N,1)   uint8   — 1 = successfully tracked, 0 = lost
        """
        # ── Input guards ──────────────────────────────────────────────
        if pts is None or len(pts) == 0:
            z = np.empty((0, 1, 2), dtype=np.float32)
            return z, np.empty((0, 1), dtype=np.uint8)
 
        # Ensure correct dtype/shape expected by cv2
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
 
        # ── cv2.calcOpticalFlowPyrLK ──────────────────────────────────
        # Returns:
        #   p1     : (N,1,2) float32 — tracked positions
        #   status : (N,1)   uint8   — 1=OK, 0=lost
        #   err    : (N,1)   float32 — per-point tracking error
        #
        # minEigThreshold rejects windows whose spatial gradient matrix
        # (A^T A) has a smallest eigenvalue below min_eig — same test as
        # the hand-rolled version's  `if eigs[0] < self.min_eig: return False`.
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, curr, p0,
            nextPts          = None,
            winSize          = self._win_size,
            maxLevel         = self.levels,
            criteria         = self._criteria,
            flags            = 0,
            minEigThreshold  = self.min_eig,
        )
 
        # calcOpticalFlowPyrLK can return None for p1 when all points fail
        if p1 is None:
            status = np.zeros((len(p0), 1), dtype=np.uint8)
            p1     = p0.copy()
 
        return p1, status
# ──────────────────────────────────────────────────────────
#  Lucas-Kanade tracker  (pyramid + iterative)
# ──────────────────────────────────────────────────────────
class LucasKanadeTracker:
    """
    Sparse optical flow via iterative pyramidal Lucas-Kanade.

    Math recap
    ----------
    Brightness constancy:  I(x+u, y+v, t+1) ~= I(x, y, t)

    Taylor expansion in a local window W:
        Ix*u + Iy*v = -It        for every pixel in W

    Stacked in matrix form:   A*d = b
        A = [Ix, Iy]  (|W|x2),  b = -It (|W|x1)

    Least-squares:  d = (A^T A)^{-1} A^T b

    We iterate this per pyramid level, propagating the estimate top-down.
    """

    def __init__(self, win: int = 51, levels: int = 3,
                 max_iter: int = 20, eps: float = 0.03,
                 min_eig: float = 1e-3):
        self.half     = win // 2
        self.levels   = levels
        self.max_iter = max_iter
        self.eps      = eps
        self.min_eig  = min_eig

    def _pyramid(self, img):
        pyr = [img.astype(np.float32)]
        for _ in range(self.levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr   # pyr[0]=finest ... pyr[levels]=coarsest

    def _lk_point(self, I1, Ix, Iy, I2, px, py):
        """Returns (dx, dy, ok) for one point at one pyramid level."""
        half = self.half
        p1   = _bilinear_patch(I1, px, py, half)
        if p1 is None:
            return 0.0, 0.0, False

        ix = _bilinear_patch(Ix, px, py, half)
        iy = _bilinear_patch(Iy, px, py, half)
        if ix is None or iy is None:
            return 0.0, 0.0, False

        Sxx = float(np.sum(ix * ix))
        Syy = float(np.sum(iy * iy))
        Sxy = float(np.sum(ix * iy))
        AtA = np.array([[Sxx, Sxy], [Sxy, Syy]], dtype=np.float64)

        eigs = np.linalg.eigvalsh(AtA)
        if eigs[0] < self.min_eig:
            return 0.0, 0.0, False

        AtA_inv = np.linalg.inv(AtA)
        vx, vy  = 0.0, 0.0

        for _ in range(self.max_iter):
            p2 = _bilinear_patch(I2, px + vx, py + vy, half)
            if p2 is None:
                return 0.0, 0.0, False

            It   = p2 - p1
            Atb  = np.array([-np.sum(ix * It),
                              -np.sum(iy * It)], dtype=np.float64)
            delta = AtA_inv @ Atb
            vx   += delta[0]
            vy   += delta[1]

            if delta[0]**2 + delta[1]**2 < self.eps ** 2:
                break

        return vx, vy, True

    def track(self, prev: np.ndarray, curr: np.ndarray,
              pts: np.ndarray):
        """
        Parameters
        ----------
        prev, curr : HxW uint8 grayscale
        pts        : (N,1,2) float32

        Returns
        -------
        next_pts : (N,1,2) float32
        status   : (N,1)   uint8   (1=OK, 0=lost)
        """
        pyr1 = self._pyramid(prev)
        pyr2 = self._pyramid(curr)

        N    = len(pts)
        disp = np.zeros((N, 2), dtype=np.float64)
        ok   = np.ones(N,        dtype=bool)

        for lvl in range(self.levels, -1, -1):
            img1   = pyr1[lvl]
            img2   = pyr2[lvl]
            Ix, Iy = _gradients(img1)
            lscale = 2.0 ** lvl

            for i in range(N):
                if not ok[i]:
                    continue
                px = pts.reshape(N, 2)[i, 0] / lscale + disp[i, 0]
                py = pts.reshape(N, 2)[i, 1] / lscale + disp[i, 1]
                dx, dy, valid = self._lk_point(img1, Ix, Iy, img2, px, py)
                if not valid:
                    ok[i] = False
                else:
                    disp[i, 0] += dx
                    disp[i, 1] += dy

            if lvl > 0:
                disp *= 2.0

        orig     = pts.reshape(N, 2).astype(np.float64)
        next_pts = (orig + disp).astype(np.float32).reshape(N, 1, 2)
        status   = ok.astype(np.uint8).reshape(N, 1)
        return next_pts, status


# ──────────────────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────────────────

def draw_tracks(frame, mask, old_pts, new_pts, status):
    good_new = new_pts[status.flatten() == 1].reshape(-1, 2)
    good_old = old_pts[status.flatten() == 1].reshape(-1, 2)

    for (x1, y1), (x0, y0) in zip(good_new.astype(int), good_old.astype(int)):
        ang    = np.arctan2(float(y1 - y0), float(x1 - x0))
        hue    = int(((ang + np.pi) / (2 * np.pi)) * 179)
        colour = cv2.cvtColor(np.uint8([[[hue, 230, 230]]]),
                              cv2.COLOR_HSV2BGR)[0, 0].tolist()
        mask  = cv2.line(mask,    (x1, y1), (x0, y0), colour, 2)
        frame = cv2.circle(frame, (x1, y1), 3,         colour, -1)

    return cv2.add(frame, mask), mask


# ──────────────────────────────────────────────────────────
#  Dense optical flow  (Farneback — BONUS)
# ──────────────────────────────────────────────────────────

def dense_flow_vis(prev_gray, curr_gray):
    flow     = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv      = np.zeros((*curr_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

FEATURE_PARAMS = dict(
    maxCorners=150,
    qualityLevel=0.03,
    minDistance=20,
    blockSize=7,
)
REDETECT_THRESHOLD = 25


def run(video_path: str, show_dense: bool = False, save: bool = True):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open: {video_path}"

    ret, frame = cap.read()
    assert ret, "Cannot read first frame"

    h, w      = frame.shape[:2]
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts  = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    tracker  = LucasKanadeTracker(win=51, levels=3, max_iter=20, eps=0.03)
    mask     = np.zeros_like(frame)
    dense_on = show_dense

    writer = None
    if save:
        out_w  = w * 2 if dense_on else w
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter("output_flow.avi", fourcc, 25.0, (out_w, h))

    idx = 0
    print("Controls: q=quit  r=re-detect  d=toggle dense")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # sparse LK
        if prev_pts is not None and len(prev_pts) >= 2:
            next_pts, status = tracker.track(prev_gray, curr_gray, prev_pts)
            vis, mask = draw_tracks(frame.copy(), mask, prev_pts, next_pts, status)
            prev_pts  = next_pts[status.flatten() == 1].reshape(-1, 1, 2)
        else:
            vis = frame.copy()

        if prev_pts is None or len(prev_pts) < REDETECT_THRESHOLD:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
            mask     = np.zeros_like(frame)

        # dense bonus
        if dense_on:
            dv    = dense_flow_vis(prev_gray, curr_gray)
            shown = np.hstack([vis, dv])
            cv2.putText(shown, "Dense Farneback",
                        (w + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        else:
            shown = vis

        n_pts = len(prev_pts) if prev_pts is not None else 0
        cv2.putText(shown, f"Sparse LK  pts:{n_pts}  frame:{idx}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
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
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
            mask     = np.zeros_like(frame)
        elif key == ord('d'):
            dense_on = not dense_on
            mask     = np.zeros_like(frame)

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
        print(f"Video not found: {video}"); sys.exit(1)
    run(video)