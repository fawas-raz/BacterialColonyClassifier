import gradio as gr

from model import model1, model2, predict_image


import gradio as gr

def gradio_predict(image, selected_model):

    if selected_model == "Model 1":
        model = model1
    else:
        model = model2

    prediction, confidence = predict_image(image, model)

    return prediction, f"{confidence * 100:.2f}%"


with gr.Blocks() as demo:

    gr.Markdown("# Bacterial Colony Classifier")

    image_input = gr.Image(
        type="pil",
        label="Upload Colony Image"
    )

    model_input = gr.Dropdown(
        choices=["Model 1", "Model 2"],
        value="Model 1",
        label="Select Model"
    )

    submit_button = gr.Button("Submit")

    prediction_output = gr.Textbox(
        label="Prediction",
        visible=False
    )

    confidence_output = gr.Textbox(
        label="Confidence",
        visible=False
    )

    def show_results(image, selected_model):
        prediction, confidence = gradio_predict(image, selected_model)
        return (
            gr.update(value=prediction, visible=True),
            gr.update(value=confidence, visible=True)
        )

    submit_button.click(
        fn=show_results,
        inputs=[image_input, model_input],
        outputs=[prediction_output, confidence_output]
    )

demo.launch()