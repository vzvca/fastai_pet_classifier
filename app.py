import gradio as gr
from fastai.vision.all import *
import skimage

def is_cat(x): return x[0].isupper()

# -- model construit
learn = load_learner('model.pkl')

#labels = learn.dls.vocab
labels = [ "dog", "cat" ]

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
examples = ['cat.jpeg','puppy-dog.jpg']

gr.Interface(fn=predict,inputs=gr.Image(width=512, height=512),outputs=gr.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples).launch()
