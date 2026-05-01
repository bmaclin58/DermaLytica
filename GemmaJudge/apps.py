from django.apps import AppConfig


class GemmaJudgeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'GemmaJudge'

    def ready(self):
        from .services.runtime import bootstrap_runtime

        bootstrap_runtime(force=False)
