from django.shortcuts import render, redirect
from django.template import RequestContext
from django.contrib import messages
from transformers import pipeline

# Create your views here.
def home(request):
    return render(request, "home.html")

def register_input(request):
    
    text_classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
    
    text = request.POST['text_area']
    
    output_classifier = text_classifier(text)
    label_classfier = output_classifier[0]['label']
    score_classifier = output_classifier[0]['score']
    
    output = f'The text is classified as: {label_classfier.capitalize()} with a score of {score_classifier}'
    messages.success(request, output)
    return redirect('/', RequestContext(request))