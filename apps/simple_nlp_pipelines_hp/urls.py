from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('register_input/', views.register_input)
]