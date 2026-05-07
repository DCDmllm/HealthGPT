import traceback

from model import HealthGPT, HealthGPT_Agent
from config import HealthGPTConfig_M3_COM, HealthGPTConfig_M3_GEN, HealthGPTConfig_L14_COM

configs = {
    "HealthGPT-M3-COM": HealthGPTConfig_M3_COM(),
    "HealthGPT-M3-GEN": HealthGPTConfig_M3_GEN(),
    "HealthGPT-L14-COM": HealthGPTConfig_L14_COM()
}

agent = HealthGPT_Agent(configs=configs, model_name=None)

# HealthGPT interface
import gradio as gr
from PIL import Image, ImageDraw

def process_input(option, model_name, text, image):
    if not text.strip():
        return gr.update(value="⚠️ Please input your question.", visible=True), None, gr.update(visible=True), gr.update(visible=False)
    try:
        if option == "Analyze Image":
            model_name = model_name + "-COM"
            try:
                agent.load_model(model_name=model_name)
                resp = agent.process(option, text, image)
            except Exception as e:
                agent.load_model(model_name=model_name)
                resp = agent.process(option, text, image)
            return resp, None, gr.update(visible=True), gr.update(visible=False)

        elif option == "Generate Image":
            model_name = model_name + "-GEN"
            try:
                agent.load_model(model_name=model_name)
                resp = agent.process(option, text, image)
            except Exception as e:
                agent.load_model(model_name=model_name)
                resp = agent.process(option, text, image)
            return None, resp, gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        print(traceback.format_exc())
        return gr.update(value=f"⚠️ {e.args[0]}", visible=True), None, gr.update(visible=True), gr.update(visible=False)


with gr.Blocks() as demo:
    # gr.Markdown("# 🖼️ HealthGPT")
    gr.Markdown("<h1 style='text-align: center; color: #333;'>🖼️ HealthGPT</h1>")

    # Option A / B
    with gr.Row():
        option = gr.Radio(["Analyze Image", "Generate Image"], label="🔍Choose the task", value="Analyze Image", interactive=True)
        model_name = gr.Radio(["HealthGPT-M3", "HealthGPT-L14"], label="🧠Choose the model", value="HealthGPT-M3", interactive=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔹 Input")
            text_input = gr.Textbox(label="Question", placeholder="Text here...", lines=3, value="Could you explain what this mass in the MRI means for my health? Is it very serious?")
            image_input = gr.Image(type="pil", label="Upload an image...")

        with gr.Column():
            gr.Markdown("### 🔹 Output")
            process_button = gr.Button("🚀 Process", variant="primary")
            text_output = gr.Textbox(label="HealthGPT Answer", visible=True, lines=20)
            image_output = gr.Image(label="Generated Image", visible=False)

    process_button.click(
        process_input,
        inputs=[option, model_name, text_input, image_input],
        outputs=[text_output, image_output, text_output, image_output]  # 用 gr.update() 代替 bool
    )

    gr.Markdown("""### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.""")

    demo.css = """footer {display: none !important;}"""

# Start Gradio website
demo.launch(server_name="0.0.0.0", server_port=5011, show_api=False)
