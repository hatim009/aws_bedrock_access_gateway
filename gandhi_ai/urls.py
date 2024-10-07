from django.urls import path
from .views import models, chat


urlpatterns = [
    path('models', models),
    path('chat/completions', chat)
]