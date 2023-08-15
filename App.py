!pip install gradio
!pip install transformers
!pip3 install torch torchvision torchaudio
#setting up hugging face pipeline
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
#model function for gradio

def func(utterance):
  return classifier(utterance)
#getting gradio library
import gradio as gr
descriptions = "This is an AI sentiment analyzer which checks and gets the emotions in a particular utterance. Just put in a sentence and you'll get the probable emotions behind that sentence"

app = gr.Interface(fn=func, inputs="text", outputs="text", title="Sentiment Analayser", description=descriptions)
app.launch('share=True')
