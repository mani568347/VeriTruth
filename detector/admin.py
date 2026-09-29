from django.contrib import admin
from .models import Source, NewsLog, AnalysisHistory

admin.site.register(Source)
admin.site.register(NewsLog)
admin.site.register(AnalysisHistory)