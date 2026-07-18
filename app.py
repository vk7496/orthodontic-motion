import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import os

from image_alignment import align_images
from landmark_morph import generate_landmark_morph_video
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="Orthodontic Motion",
    page_icon="🦷",
    layout="wide"
)


st.title("🦷 Orthodontic Motion")

st.write(
    "AI Orthodontic Tooth Movement Visualization"
)

st.divider()


before_file = st.file_uploader(
    "Upload BEFORE image",
    type=["jpg", "jpeg", "png"]
)

after_file = st.file_uploader(
    "Upload AFTER image",
    type=["jpg", "jpeg", "png"]
)


def draw_markers(pil_img, points, color=(255, 60, 60)):
    """Return a copy of pil_img with numbered circles at the given points."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    r = max(6, img.width // 120)
    for i, (x, y) in enumerate(points):
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        draw.text((x + r + 2, y - r), str(i + 1), fill=color)
    return img


if before_file and after_file:

    before_image = Image.open(before_file).convert("RGB")
    after_image = Image.open(after_file).convert("RGB")

    before = np.array(before_image)
    after = np.array(after_image)

    st.subheader("Original Images")

    col1, col2 = st.columns(2)
    with col1:
        st.image(before, caption="BEFORE", use_container_width=True)
    with col2:
        st.image(after, caption="AFTER", use_container_width=True)

    st.divider()

    if "aligned_after" not in st.session_state:
        st.session_state.aligned_after = None
    if "before_pts" not in st.session_state:
        st.session_state.before_pts = []
    if "after_pts" not in st.session_state:
        st.session_state.after_pts = []

    if st.button("🔄 Align Images", use_container_width=True):
        with st.spinner("Aligning dental images..."):
            aligned_after = align_images(before, after)
        st.session_state.aligned_after = aligned_after
        st.session_state.before_pts = []
        st.session_state.after_pts = []
        st.success("Images aligned successfully!")

    if st.session_state.aligned_after is not None:

        aligned_after = st.session_state.aligned_after

        st.subheader("Alignment Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.image(before, caption="BEFORE", use_container_width=True)
        with col2:
            st.image(aligned_after, caption="ALIGNED AFTER", use_container_width=True)

        st.divider()

        st.subheader("📍 Mark Matching Points")
        st.write(
            "Click the **same tooth landmark** on both images, one pair at a time "
            "(e.g. midline notch, canine tips, last visible molar on each side). "
            "Aim for **6–10 pairs** spread across the arch for a convincing result."
        )

        before_pts = st.session_state.before_pts
        after_pts = st.session_state.after_pts

        # whichever list is shorter tells us which image should capture the next click
        waiting_on = "before" if len(before_pts) <= len(after_pts) else "after"
        next_index = max(len(before_pts), len(after_pts)) + 1

        col1, col2 = st.columns(2)

        with col1:
            st.caption(
                f"👉 Click point #{next_index} here" if waiting_on == "before"
                else f"BEFORE — {len(before_pts)} point(s) marked"
            )
            before_marked = draw_markers(Image.fromarray(before), before_pts)
            click = streamlit_image_coordinates(before_marked, key="before_click")
            if click is not None and waiting_on == "before":
                pt = (click["x"], click["y"])
                if not before_pts or before_pts[-1] != pt:
                    before_pts.append(pt)
                    st.rerun()

        with col2:
            st.caption(
                f"👉 Now click the SAME spot (point #{next_index}) here" if waiting_on == "after"
                else f"ALIGNED AFTER — {len(after_pts)} point(s) marked"
            )
            after_marked = draw_markers(Image.fromarray(aligned_after), after_pts)
            click2 = streamlit_image_coordinates(after_marked, key="after_click")
            if click2 is not None and waiting_on == "after":
                pt = (click2["x"], click2["y"])
                if not after_pts or after_pts[-1] != pt:
                    after_pts.append(pt)
                    st.rerun()

        col_u, col_r = st.columns(2)
        with col_u:
            if st.button("↩️ Undo last point", use_container_width=True):
                if len(before_pts) == len(after_pts) and before_pts:
                    before_pts.pop()
                    after_pts.pop()
                elif len(before_pts) > len(after_pts):
                    before_pts.pop()
                elif len(after_pts) > len(before_pts):
                    after_pts.pop()
                st.rerun()
        with col_r:
            if st.button("🗑️ Reset points", use_container_width=True):
                st.session_state.before_pts = []
                st.session_state.after_pts = []
                st.rerun()

        n_pairs = min(len(before_pts), len(after_pts))
        st.write(f"**{n_pairs} matched point pair(s)** ready.")

        st.divider()

        st.subheader("🎬 Motion Video")

        col_a, col_b = st.columns(2)
        with col_a:
            duration_choice = st.select_slider(
                "Video length",
                options=["Quick (6s)", "Standard (10s)", "Long (14s)"],
                value="Standard (10s)"
            )
        with col_b:
            fps = st.select_slider("Smoothness (fps)", options=[24, 30, 60], value=30)

        duration_map = {
            "Quick (6s)":     dict(hold_before_s=1.0, move_s=3.0, dissolve_s=0.5, hold_after_s=1.5),
            "Standard (10s)": dict(hold_before_s=2.0, move_s=4.0, dissolve_s=1.0, hold_after_s=3.0),
            "Long (14s)":     dict(hold_before_s=3.0, move_s=6.0, dissolve_s=1.0, hold_after_s=4.0),
        }

        if n_pairs < 5:
            st.info("Mark at least 5 matching point pairs to enable video generation.")
        else:
            if st.button("🎬 Generate Motion Video", use_container_width=True, type="primary"):
                progress_bar = st.progress(0.0, text="Building motion video...")

                def _cb(frac):
                    progress_bar.progress(min(frac, 1.0), text=f"Rendering frames... {int(frac*100)}%")

                out_path = os.path.join(tempfile.gettempdir(), "orthodontic_motion.mp4")

                generate_landmark_morph_video(
                    before,
                    aligned_after,
                    before_pts[:n_pairs],
                    after_pts[:n_pairs],
                    out_path,
                    fps=fps,
                    progress_cb=_cb,
                    **duration_map[duration_choice],
                )

                progress_bar.progress(1.0, text="Done!")
                st.success("Motion video generated!")
                st.video(out_path)

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download MP4",
                        data=f,
                        file_name="orthodontic_motion.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                )
