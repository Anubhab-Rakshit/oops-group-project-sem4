# Use an official Python slim image for a smaller footprint
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache

# Set work directory
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Pre-download the SBERT model into the image
RUN python download_model.py

# Create a volume for input/output files and cache
VOLUME ["/data", "/app/.hf_cache"]

# Expose port for the API
EXPOSE 7860

# Command to run the FastAPI server
# We bind to 0.0.0.0 so it's accessible outside the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
