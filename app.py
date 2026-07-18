import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Orthodontic Motion",
    page_icon="🦷",
    layout="centered"
)

st.title("🦷 Orthodontic Motion")
st.write("AI Orthodontic Tooth Movement Visualization")

st.divider()

before_image = st.file_uploader(
    "Upload BEFORE image",
    type=["jpg", "jpeg", "png"]
)

after_image = st.file_uploader(
    "Upload AFTER image",
    type=["jpg", "jpeg", "png"]
)

if before_image and after_image:

    before = Image.open(before_image)
    after = Image.open(after_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before")
        st.image(before, use_container_width=True)

    with col2:
        st.subheader("After")
        st.image(after, use_container_width=True)

    st.divider()

    if st.button("🚀 Generate Simulation", use_container_width=True):
        st.success("Images uploaded successfully!")
        st.info("Tooth movement engine will be added in the next step.")
