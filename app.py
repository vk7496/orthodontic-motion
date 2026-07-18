import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

from image_alignment import align_images
from morph_video import generate_morph_video


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


if before_file and after_file:

    before_image = Image.open(
        before_file
    ).convert("RGB")

    after_image = Image.open(
        after_file
    ).convert("RGB")


    before = np.array(
        before_image
    )

    after = np.array(
        after_image
    )


    st.subheader("Original Images")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            before,
            caption="BEFORE",
            use_container_width=True
        )

    with col2:

        st.image(
            after,
            caption="AFTER",
            use_container_width=True
        )


    st.divider()

    # keep aligned result across reruns (so the video button below doesn't
    # need to re-run alignment every time Streamlit reruns the script)
    if "aligned_after" not in st.session_state:
        st.session_state.aligned_after = None

    if st.button(
        "🔄 Align Images",
        use_container_width=True
    ):

        with st.spinner(
            "Aligning dental images..."
        ):

            aligned_after = align_images(
                before,
                after
            )

        st.session_state.aligned_after = aligned_after

        st.success(
            "Images aligned successfully!"
        )

    if st.session_state.aligned_after is not None:

        aligned_after = st.session_state.aligned_after

        st.subheader(
            "Alignment Preview"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                before,
                caption="BEFORE",
                use_container_width=True
            )

        with col2:

            st.image(
                aligned_after,
                caption="ALIGNED AFTER",
                use_container_width=True
            )

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
            fps = st.select_slider(
                "Smoothness (fps)",
                options=[24, 30, 60],
                value=30
            )

        duration_map = {
            "Quick (6s)":    dict(hold_before_s=1.0, move_s=3.0, dissolve_s=0.5, hold_after_s=1.5),
            "Standard (10s)": dict(hold_before_s=2.0, move_s=4.0, dissolve_s=1.0, hold_after_s=3.0),
            "Long (14s)":    dict(hold_before_s=3.0, move_s=6.0, dissolve_s=1.0, hold_after_s=4.0),
        }

        if st.button("🎬 Generate Motion Video", use_container_width=True, type="primary"):

            progress_bar = st.progress(0.0, text="Building motion video...")

            def _cb(frac):
                progress_bar.progress(min(frac, 1.0), text=f"Rendering frames... {int(frac*100)}%")

            out_path = os.path.join(tempfile.gettempdir(), "orthodontic_motion.mp4")

            generate_morph_video(
                before,
                aligned_after,
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
