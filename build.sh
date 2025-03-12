#!/usr/bin/env bash
# Exit on error
set -o errexit

pip install -r requirements.txt

# Ensure the model directory exists
mkdir -p DermaLytica/Prediction_Model/AI_Models

# Download the ML model from Google Cloud Storage
wget -O DermaLytica/Prediction_Model/AI_Models/KERAS_model.tflite "https://storage.googleapis.com/dermalyticsdrive/models/KERAS_model.tflite"
