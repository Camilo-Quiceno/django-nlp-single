from django.shortcuts import render, redirect
from django.template import RequestContext
from django.contrib import messages
from transformers import pipeline

# Create your views here.
def home(request):
    return render(request, "home.html")

def register_input(request):
        
    text = request.POST['text_area']
    error = False
    
    match request.session['pipeline_selected']:
        case "1":
            try:
                summarizer = pipeline(
                    "summarization", model="t5-base", tokenizer="t5-base", framework="tf"
                )
                
                output_summarizer = summarizer(text, min_length=5, max_length=35)[0]['summary_text']
                output = f'The text summarize is: {output_summarizer}'
            except:
                error = True
                error_text = "Something wrong, try again" 
            
        case "2":
            try:
                question = text.split(",")[0]
                context = text.split(",")[1]
            
                question_answering = pipeline(model="deepset/roberta-base-squad2")
                
                output_question_answering = question_answering(
                                    question=question,
                                    context=context,
                                )
                answer_qa = output_question_answering['answer']
                score_qa = output_question_answering['score']
                output = f'The answer of the question is: {answer_qa} with a score of: {round(score_qa, 2)}'
            except:
                error = True
                error_text = "Incorrect Input, try again."          
            
        case "3":
            try:
                entity_classifier = pipeline(
                    model="dslim/bert-base-NER-uncased", aggregation_strategy="simple"
                )
                
                sentence = text
                entities = entity_classifier(sentence)
                output = ""
                for entity in entities:
                        entity_group = entity['entity_group']
                        word = entity['word']
                        single_output = f'- The entity group of the word {word} is {entity_group}\n'
                        output += single_output
            except:
                error = True
                error_text = "Something wrong, try again." 
                    
        case "4":
            try:
                pos_tagger = pipeline(
                    model="vblagoje/bert-english-uncased-finetuned-pos",
                    aggregation_strategy="simple",
                )
                
                taggings = pos_tagger(text)
                output = ""
                for tagging in taggings:
                        entity_group = tagging['entity_group']
                        word = tagging['word']
                        single_output = f'The entity group of the word {word} is {entity_group}\n'
                        output += single_output
            except:
                error = True
                error_text = "Something wrong, try again." 
                    
            
        case "5":
            try:
                text_classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
                
                output_classifier = text_classifier(text)
                label_classfier = output_classifier[0]['label']
                score_classifier = output_classifier[0]['score']
                output = f'The text is classified as: {label_classfier.capitalize()} with a score of {score_classifier}'
            except:
                error = True
                error_text = "Something wrong, try again." 
            
        case "6":
            try:
                text_generator = pipeline(model="gpt2")
                
                generated_text = text_generator(text, do_sample=False)[0]['generated_text']
                output = f'The text generated is: {generated_text}'
            except:
                error = True
                error_text = "Something wrong, try again." 
            
        case "7":
            try:
                en_fr_translator = pipeline("translation_en_to_fr", model='t5-small')
                
                output_translator = en_fr_translator(text)
                text_translated = output_translator[0]['translation_text']
                output = f'The text translated is: {text_translated}'
            except:
                error = True
                error_text = "Something wrong, try again." 
            
        case _:
            text = "Default"
            
    if error:
        messages.error(request, error_text, extra_tags='error')
    else:
        messages.success(request, output, extra_tags='pipeline_output')
    
    
    return redirect('/', RequestContext(request))

def form_selection(request):
    
    form_select = request.POST['form_select']
    
    match form_select:
        case "1":
            text = "The input to this task is a corpus of text and the model will output a summary of it based on the expected length mentioned in the parameters. Here, we have kept minimum length as 5 and maximum length as 30. \n\nExample:\n\nParents need to know that Top Gun is a blockbuster 1980s action thriller starring Tom Cruise that's chock full of narrow escapes, chases, and battles. But there are also violent and upsetting scenes, particularly the death of a main character, which make it too intense for younger kids. There's also one graphic-for-its-time sex scene (though no explicit nudity) and quite a few shirtless men in locker rooms and, in one iconic sequence, on a beach volleyball court. Winning is the most important thing to all the pilots, who try to intimidate one another with plenty of posturing and banter -- though when push comes to shove, loyalty and friendship have important roles to play, too. While sexism is noticeable and almost all characters are men, two strong women help keep some of the objectification in check."
        case "2":
            text = "In this task, we provide a question and a context ( Separate the question and the context by a comma ',' ). The model will choose the answer from the context based on the highest probability score. It also provides the starting and ending positions of the text. \n\nExample:\n\nWhere do I work?, I work as a Data Scientist at a lab in University of Montreal. I like to develop my own algorithms."
        case "3":
            text = "Named Entity Recognition deals with identifying and classifying the words based on the names of persons, organizations, locations and so on. The input is basically a sentence and the model will determine the named entity along with its category and its corresponding location in the text. \n\nExample:\n\nJohn and Jane like to travel around Spain by train."
        case "4":
            text = "PoS Tagging is useful to classify the text and provide its relevant parts of speech such as whether a word is a noun, pronoun, verb and so on. The model returns PoS tagged words along with their probability scores and respective locations. \n\nExample:\n\nI am an artist and I live in Dublin"
        case "5":
            text = "We will perform sentiment analysis and classify the text based on the tone. \n\nExample:\n\nThis movie is my favorite!"
        case "6":
            text = "Text will be generated base on your input. \n\nExample:\n\nIf it is sunny today then "
        case "7":
            text = "Here, we will translate the language of text from one language to another. English -> French. \n\nExample:\n\nHi, How are you?"
        case _:
            text = "Default"
    
    request.session['pipeline_selected'] = form_select
    context = {'selected_value': form_select}
    messages.success(request, text, extra_tags='form_selection')
    
    return render(request, 'home.html', context)