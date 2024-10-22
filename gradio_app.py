import gradio as gr
from inference import inference_and_run
import spaces
import os
import re
import shutil

model_name = 'Ferret-UI'
cur_dir = os.path.dirname(os.path.abspath(__file__))

@spaces.GPU()
def inference_with_gradio(chatbot, image, prompt, model_path, box=None):
    dir_path = os.path.dirname(image)
    # image_path = image
    # Define the directory where you want to save the image (current directory)
    filename = os.path.basename(image)
    dir_path = "./"

    # Create the new path for the file (in the current directory)
    image_path = os.path.join(dir_path, filename)
    shutil.copy(image, image_path)
    print("filename path: ", filename)
    if "gemma" in model_path.lower():
        conv_mode = "ferret_gemma_instruct"
    else:
        conv_mode = "ferret_llama_3"
    
    # inference_text = inference_and_run(
    #     image_path=image_path,
    #     prompt=prompt,
    #     conv_mode=conv_mode,
    #     model_path=model_path,
    #     box=box
    # )
    inference_text = inference_and_run(
        image_path=filename, # double check this
        image_dir=dir_path,
        prompt=prompt,
        model_path="jadechoghari/Ferret-UI-Gemma2b",
        conv_mode=conv_mode,  # Default mode from the original function
        # temperature=temperature, 
        # top_p=top_p,
        # max_new_tokens=max_new_tokens,
        # stop=stop    # Assuming we want to process the image
        )
    
    # print("done, now appending", inference_text)
    # chatbot.append((prompt, inference_text))
    # return chatbot
    # Convert inference_text to string if it's not already
    if isinstance(inference_text, (list, tuple)):
        inference_text = str(inference_text[0])
        
    # Update chatbot history with new message pair
    new_history = chatbot.copy() if chatbot else []
    new_history.append((prompt, inference_text))
    return new_history

def submit_chat(chatbot, text_input):
    response = ''
    chatbot.append((text_input, response))
    return chatbot, ''

def clear_chat():
    return [], None, ""

with open(f"{cur_dir}/ferretui_icon.svg", "r", encoding="utf-8") as svg_file:
    svg_content = svg_file.read()
font_size = "2.5em"
svg_content = re.sub(r'(<svg[^>]*)(>)', rf'\1 height="{font_size}" style="vertical-align: middle; display: inline-block;"\2', svg_content)
html = f"""
<p align="center" style="font-size: {font_size}; line-height: 1;">
    <span style="display: inline-block; vertical-align: middle;">{svg_content}</span>
    <span style="display: inline-block; vertical-align: middle;">{model_name}</span>
</p>
<center><font size=3><b>{model_name}</b> Demo: Upload an image, provide a prompt, and get insights using advanced AI models. <a href='https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b'>😊 Huggingface</a></font></center>
"""

latex_delimiters_set = [{
        "left": "\\(",
        "right": "\\)",
        "display": False 
    }, {
        "left": "\\begin{equation}",
        "right": "\\end{equation}",
        "display": True 
    }, {
        "left": "\\begin{align}",
        "right": "\\end{align}",
        "display": True
    }]

# Set up UI components
image_input = gr.Image(label="Upload Image", type="filepath", height=350)
text_input = gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt")
model_dropdown = gr.Dropdown(choices=[
    "jadechoghari/Ferret-UI-Gemma2b",
    "jadechoghari/Ferret-UI-Llama8b",
], label="Model Path", value="jadechoghari/Ferret-UI-Gemma2b")

bounding_box_input = gr.Textbox(placeholder="Optional bounding box (x1, y1, x2, y2)", label="Bounding Box (optional)")
chatbot = gr.Chatbot(label="Chat with Ferret-UI", height=400, show_copy_button=True, latex_delimiters=latex_delimiters_set)

with gr.Blocks(title=model_name, theme=gr.themes.Ocean()) as demo:
    gr.HTML(html)
    with gr.Row():
        with gr.Column(scale=3):
            # gr.Examples(
            #     examples=[
            #         ["appstore_reminders.png", "Describe the image in details", "jadechoghari/Ferret-UI-Gemma2b", None],
            #         ["appstore_reminders.png", "What's inside the selected region?", "jadechoghari/Ferret-UI-Gemma2b", "189, 906, 404, 970"],
            #         ["appstore_reminders.png", "Where is the Game Tab?", "jadechoghari/Ferret-UI-Gemma2b", None],
            #     ],
            #     inputs=[image_input, text_input, model_dropdown, bounding_box_input]
            # )
            image_input.render()
            text_input.render()
            model_dropdown.render()
            bounding_box_input.render()
        with gr.Column(scale=7):
            chatbot.render()
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

    send_click_event = send_btn.click(
        inference_with_gradio, [chatbot, image_input, text_input, model_dropdown, bounding_box_input], chatbot
    ).then(submit_chat, [chatbot, text_input], [chatbot, text_input])
    submit_event = text_input.submit(
        inference_with_gradio, [chatbot, image_input, text_input, model_dropdown, bounding_box_input], chatbot
    ).then(submit_chat, [chatbot, text_input], [chatbot, text_input])
    
    clear_btn.click(clear_chat, outputs=[chatbot, image_input, text_input, bounding_box_input])

demo.launch()
