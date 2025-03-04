import os

from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'DermaLytica', 'Prediction_Model', 'AI_Models', 'derma_model_keras.keras')
AGE_MEAN = 57.70533017
AGE_STD = 14.11323567
IMAGE_SIZE = (224, 224)
OPTIMAL_THRESHOLD = 0.2494
