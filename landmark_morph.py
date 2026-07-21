"""
landmark_morph.py

Landmark-driven morph: instead of guessing motion with dense optical flow
(which fails when teeth look too similar to each other / to gum), the user
clicks a handful of corresponding points on BEFORE and AFTER (same tooth /
same landmark, in the same order). We then warp BEFORE toward AFTER using
a Thin Plate Spline (TPS) computed from those points, interpolating the
point positions smoothly over time. This guarantees the pixels the user
identified as "this tooth" actually travel to where they told us the same
tooth ends up.
"""

import cv2
import numpy as np


def _boundary_anchors(w, h, n_per_side=3):
    """Fixed anchor points around the image border so the background /
    frame doesn't get dragged around by the interior warp."""
    pts = []
    for i in range(n_per_side):
        x = i * (w - 1) / (n_per_side - 1)
        pts.append((x, 0))
        pts.append((x, h - 1))
    for i in range(1, n_per_side - 1):
        y = i * (h - 1) / (n_per_side - 1)
        pts.append((0, y))
        pts.append((w - 1, y))
    return pts


def build_matches(before_pts, after_pts, w, h):
    """
    before_pts, after_pts: lists of (x, y) in the SAME order (point i in
    before corresponds to point i in after).
    Returns (src_pts, dst_pts) as Nx2 float32 arrays, including fixed
    boundary anchors.
    """
    anchors = _boundary_anchors(w, h)
    src = list(before_pts) + anchors
    dst = list(after_pts) + anchors
    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)


def tps_warp(img_bgr, src_pts, dst_pts):
    """Warp img_bgr so that src_pts land on dst_pts, via Thin Plate Spline."""
    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    src_shape = src_pts.reshape(1, -1, 2)
    dst_shape = dst_pts.reshape(1, -1, 2)
    # NOTE: source shape first, target shape second -- passing these
    # reversed silently produces a near-identity warp (points barely move).
    tps.estimateTransformation(src_shape, dst_shape, matches)
    warped = tps.warpImage(img_bgr)
    return warped


def _ease_in_out(x):
    return x * x * (3 - 2 * x)


def generate_landmark_morph_video(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    before_pts,
    after_pts,
    out_path: str,
    fps: int = 30,
    hold_before_s: float = 2.0,
    move_s: float = 4.0,
    dissolve_s: float = 1.0,
    hold_after_s: float = 3.0,
    progress_cb=None,
):
    import imageio.v2 as imageio

    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)

    H0, W0 = before_bgr.shape[:2]
    H, W = H0 - (H0 % 2), W0 - (W0 % 2)
    before_bgr = before_bgr[:H, :W]
    after_bgr = after_bgr[:H, :W]

    src_pts, dst_pts = build_matches(before_pts, after_pts, W, H)

    n_hold_before = max(1, int(hold_before_s * fps))
    n_move = max(2, int(move_s * fps))
    n_dissolve = max(1, int(dissolve_s * fps))
    n_hold_after = max(1, int(hold_after_s * fps))
    total = n_hold_before + n_move + n_dissolve + n_hold_after

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                 quality=8, macro_block_size=None)
    done = 0

    def emit_bgr(frame_bgr):
        nonlocal done
        writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        done += 1
        if progress_cb:
            progress_cb(done / total)

    for _ in range(n_hold_before):
        emit_bgr(before_bgr)

    warped_frames_cache = {}
    for i in range(n_move):
        t = _ease_in_out(i / (n_move - 1))
        interp_pts = src_pts + (dst_pts - src_pts) * t
        warped = tps_warp(before_bgr, src_pts, interp_pts)
        emit_bgr(warped)

    fully_warped = tps_warp(before_bgr, src_pts, dst_pts)
    for i in range(1, n_dissolve + 1):
        a = _ease_in_out(i / n_dissolve)
        blended = cv2.addWeighted(fully_warped, 1 - a, after_bgr, a, 0)
        emit_bgr(blended)

    for _ in range(n_hold_after):
        emit_bgr(after_bgr)

    writer.close()
    return out_path
