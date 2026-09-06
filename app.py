import streamlit as st
from PIL import Image

from model import model1, model2, predict_image


st.title("Bacterial Colony Classifier")

image_input = st.file_uploader(
    "Upload Colony Image",
    type=["jpg", "jpeg", "png"]
)

selected_model = st.selectbox(
    "Select Model",
    ["Model 1", "Model 2"]
)

if st.button("Submit"):

    if image_input is not None:

        image = Image.open(image_input)

        if selected_model == "Model 1":
            model = model1
        else:
            model = model2

        prediction, confidence = predict_image(image, model)

        st.write("### Prediction")
        st.write(prediction)

        st.write("### Confidence")
        st.write(f"{confidence * 100:.2f}%")

    else:
        st.warning("Please upload an image.")