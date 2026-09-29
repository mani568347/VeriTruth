from rest_framework import serializers
from .models import  AnalysisHistory

class PredictionSerializer(serializers.ModelSerializer):

    class Meta:

        model = AnalysisHistory

        fields = "__all__"