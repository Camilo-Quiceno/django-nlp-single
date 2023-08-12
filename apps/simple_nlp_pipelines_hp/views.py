from django.shortcuts import render, redirect
from django.template import RequestContext
from django.contrib import messages

# Create your views here.
def home(request):
    return render(request, "home.html")

def register_input(request):
    text = request.POST['text_area']
    messages.success(request, text)
    return redirect('/', RequestContext(request))