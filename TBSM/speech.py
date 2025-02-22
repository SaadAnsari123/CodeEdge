import gradio as gr
import speech_recognition as sr

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)  # Just get one audio input, not iterable
        try:
            text = recognizer.recognize_google(audio)  # Convert speech to text
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech recognition service error"

# Gradio UI
iface = gr.Interface(fn=transcribe_speech, inputs=[], outputs="text", live=True)
iface.launch()
