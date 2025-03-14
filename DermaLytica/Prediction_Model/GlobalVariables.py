def downloadModel():
    import os
    import requests
    import hashlib
    from django.conf import settings

    MODEL_URL = "https://storage.googleapis.com/dermalyticsdrive/models/KERAS_model.tflite"
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'DermaLytica', 'Prediction_Model', 'AI_Models', 'KERAS_model.tflite')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    download_needed = False

    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        download_needed = True
        print(f"Model file does not exist at {MODEL_PATH}")
    else:
        # Optional: Check file size or integrity
        try:
            # If you know the expected file size
            file_size = os.path.getsize(MODEL_PATH)
            if file_size < 1000:  # Arbitrary small size check
                print(f"Model file seems too small ({file_size} bytes), redownloading...")
                download_needed = True
        except Exception as e:
            print(f"Error checking file: {e}")
            download_needed = True

    if download_needed:
        print(f"Downloading model from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print("Model file already exists and looks valid, skipping download.")

    return MODEL_PATH

AGE_MEAN = 57.70533017
AGE_STD = 14.11323567
IMAGE_SIZE = (224, 224)
OPTIMAL_THRESHOLD = 0.2494
