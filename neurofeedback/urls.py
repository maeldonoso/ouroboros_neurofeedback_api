from django.urls import path
from .views import observed_brain_activity, predicted_brain_activity

urlpatterns = [
    path('observed_brain_activity', observed_brain_activity),
    path('predicted_brain_activity', predicted_brain_activity),
]