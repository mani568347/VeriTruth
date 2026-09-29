import os
import sys
import threading
from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'

    def ready(self):
        # Skip warmup for management commands like migrations
        skip_commands = {'makemigrations', 'migrate', 'collectstatic', 'check', 'help'}
        if any(cmd in sys.argv for cmd in skip_commands):
            return

        # Under the autoreloader, warm up only in the worker process.
        # With --noreload there is no worker, so don't skip.
        if ('runserver' in sys.argv and '--noreload' not in sys.argv
                and os.environ.get('RUN_MAIN') != 'true'):
            return

        def _warmup():
            try:
                from ml.bert_service import warmup_model
                warmup_model()
            except Exception as exc:
                print(f"[VeriTruth] Background warmup notice: {exc}")

        thread = threading.Thread(target=_warmup, daemon=True)
        thread.start()

