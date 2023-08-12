from django.db import models

# Create your models here.
class Input(models.Model):
    text = models.TextField()