# -------------------------------
# Stage 1: Build dependencies
# -------------------------------
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Stage 2: Final application image
# -------------------------------
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local /usr/local

# Copy application source code
COPY . /app/

# Download the model file
RUN mkdir -p DermaLytica/Prediction_Model/AI_Models && \
    curl -o DermaLytica/Prediction_Model/AI_Models/KERAS_model.tflite \
    https://storage.googleapis.com/dermalyticsdrive/models/KERAS_model.tflite

# Expose Cloud Run's required port
EXPOSE 8080

# Use gunicorn to start the app on the required port
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:application"]
