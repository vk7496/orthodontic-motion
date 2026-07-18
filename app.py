import streamlit as st
from PIL import Image
import numpy as np

from image_alignment import align_images


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


        st.success(
            "Images aligned successfully!"
        )


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
