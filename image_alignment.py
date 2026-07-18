import cv2
import numpy as np


def align_images(before, after):
    """
    Align AFTER image to BEFORE image using ECC image registration.
    """

    before_gray = cv2.cvtColor(
        before,
        cv2.COLOR_RGB2GRAY
    )

    after_gray = cv2.cvtColor(
        after,
        cv2.COLOR_RGB2GRAY
    )

    # Resize AFTER to BEFORE size
    after_resized = cv2.resize(
        after,
        (before.shape[1], before.shape[0])
    )

    after_gray = cv2.cvtColor(
        after_resized,
        cv2.COLOR_RGB2GRAY
    )

    # Normalize images
    before_gray = cv2.GaussianBlur(
        before_gray,
        (5, 5),
        0
    )

    after_gray = cv2.GaussianBlur(
        after_gray,
        (5, 5),
        0
    )

    # Affine transformation
    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        200,
        1e-6
    )

    try:

        _, warp_matrix = cv2.findTransformECC(
            before_gray,
            after_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria
        )

        aligned_after = cv2.warpAffine(
            after_resized,
            warp_matrix,
            (
                before.shape[1],
                before.shape[0]
            ),
            flags=cv2.INTER_LINEAR +
            cv2.WARP_INVERSE_MAP
        )

        return aligned_after

    except cv2.error:

        return after_resized
