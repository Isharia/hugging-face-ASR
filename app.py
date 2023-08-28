from transformers import pipeline
import gradio as gr
import time

#p = pipeline("automatic-speech-recognition")



model_id = "openai/whisper-small"

p = pipeline("automatic-speech-recognition", model=model_id)

def transcribe(audio, state=""):
    time.sleep(2)
    text = p(audio)["text"]
    state += text + " "
    return state, state



# Set the starting state to an empty string

gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(source="microphone", type="filepath", streaming=True),
        "state"
    ],
    outputs=[
        "textbox",
        "state"
    ],
    live=True).launch()

gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text").launch()
