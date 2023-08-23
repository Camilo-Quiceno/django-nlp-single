from django.shortcuts import render, redirect
from django.template import RequestContext
from django.contrib import messages
from transformers import pipeline

# Create your views here.
def home(request):
    return render(request, "home.html")

def register_input(request):
        
    text = request.POST['text_area']
    
    match request.session['pipeline_selected']:
        case "1":
            text = "The input to this task is a corpus of text and the model will output a summary of it based on the expected length mentioned in the parameters. Here, we have kept minimum length as 5 and maximum length as 30."
        case "2":
            text = "In this task, we provide a question and a context. The model will choose the answer from the context based on the highest probability score. It also provides the starting and ending positions of the text."
        case "3":
            text = "Named Entity Recognition deals with identifying and classifying the words based on the names of persons, organizations, locations and so on. The input is basically a sentence and the model will determine the named entity along with its category and its corresponding location in the text. "
        case "4":
            text = "PoS Tagging is useful to classify the text and provide its relevant parts of speech such as whether a word is a noun, pronoun, verb and so on. The model returns PoS tagged words along with their probability scores and respective locations."
        case "5":
            text_classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
            output_classifier = text_classifier(text)
            label_classfier = output_classifier[0]['label']
            score_classifier = output_classifier[0]['score']
            output = f'The text is classified as: {label_classfier.capitalize()} with a score of {score_classifier}'
        case "6":
            text = "Text will be generated base on your input."
        case "7":
            text = "Here, we will translate the language of text from one language to another. English -> French"
        case _:
            text = "Caso predeterminado"
    
    messages.success(request, output, extra_tags='pipeline_output')
    return redirect('/', RequestContext(request))

def form_selection(request):
    
    form_select = request.GET.get('form_select', 'form_select')
    
    match form_select:
        case "1":
            text = "The input to this task is a corpus of text and the model will output a summary of it based on the expected length mentioned in the parameters. Here, we have kept minimum length as 5 and maximum length as 30."
        case "2":
            text = "In this task, we provide a question and a context. The model will choose the answer from the context based on the highest probability score. It also provides the starting and ending positions of the text."
        case "3":
            text = "Named Entity Recognition deals with identifying and classifying the words based on the names of persons, organizations, locations and so on. The input is basically a sentence and the model will determine the named entity along with its category and its corresponding location in the text. "
        case "4":
            text = "PoS Tagging is useful to classify the text and provide its relevant parts of speech such as whether a word is a noun, pronoun, verb and so on. The model returns PoS tagged words along with their probability scores and respective locations."
        case "5":
            text = "We will perform sentiment analysis and classify the text based on the tone."
        case "6":
            text = "Text will be generated base on your input."
        case "7":
            text = "Here, we will translate the language of text from one language to another. English -> French"
        case _:
            text = "Caso predeterminado"
    
    request.session['pipeline_selected'] = form_select
    context = {'selected_value': form_select}
    messages.success(request, text, extra_tags='form_selection')
    return render(request, 'home.html', context)