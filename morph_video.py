"""
morph_video.py

Generates an AI-assisted "before -> after" orthodontic motion video from two
already-aligned images (see image_alignment.align_images).

Technique: dense optical flow (Farneback) computes a displacement field from
BEFORE to AFTER. We then:
  1. Hold on BEFORE.
  2. Warp BEFORE forward along the flow field (pure geometric movement,
     no cross-blending) so the teeth appear to physically move.
  3. In the final short window, quickly dissolve from the fully-warped
     BEFORE into the real AFTER photo (this swaps texture/material, e.g.
     a removed wire or whitened teeth, which a warp alone cannot do).
  4. Hold on AFTER.

This mirrors classic face-morph pipelines (warp-then-dissolve) rather than
a plain two-image cross-fade, which reads as real "movement" instead of a
double-exposure blend.
"""

import cv2
import numpy as np
import imageio.v2 as imageio


def _ease_in_out(x):
    return x * x * (3 - 2 * x)


def _compute_flow(before_bgr, after_bgr):
    g1 = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (7, 7), 0)
    g2 = cv2.GaussianBlur(g2, (7, 7), 0)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=6, winsize=31,
        iterations=6, poly_n=7, poly_sigma=1.5, flags=0
    )
    # smooth the field so movement is coherent rather than noisy per-pixel jitter
    flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (21, 21), 0)
    flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (21, 21), 0)
    return flow


def _warp_forward(img, flow, t, grid_x, grid_y):
    map_x = (grid_x + flow[..., 0] * t).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * t).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)


def generate_morph_video(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    out_path: str,
    fps: int = 30,
    hold_before_s: float = 2.0,
    move_s: float = 4.0,
    dissolve_s: float = 1.0,
    hold_after_s: float = 3.0,
    progress_cb=None,
):
    """
    before_rgb, after_rgb: HxWx3 RGB uint8 arrays, SAME SIZE
    (after_rgb should already be the ECC-aligned version from image_alignment.align_images)
    out_path: path to write the .mp4 file
    progress_cb: optional callable(fraction: float) for a Streamlit progress bar
    """
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)

    # libx264 requires even width/height - crop 1px off the edge if needed
    H0, W0 = before_bgr.shape[:2]
    H, W = H0 - (H0 % 2), W0 - (W0 % 2)
    before_bgr = before_bgr[:H, :W]
    after_bgr = after_bgr[:H, :W]
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)

    flow = _compute_flow(before_bgr, after_bgr)

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

    for i in range(n_move):
        t = _ease_in_out(i / (n_move - 1))
        emit_bgr(_warp_forward(before_bgr, flow, t, grid_x, grid_y))

    fully_warped = _warp_forward(before_bgr, flow, 1.0, grid_x, grid_y)
    for i in range(1, n_dissolve + 1):
        a = _ease_in_out(i / n_dissolve)
        blended = cv2.addWeighted(fully_warped, 1 - a, after_bgr, a, 0)
        emit_bgr(blended)

    for _ in range(n_hold_after):
        emit_bgr(after_bgr)

    writer.close()
    return out_path
