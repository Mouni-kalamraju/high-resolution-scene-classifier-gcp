# Use Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Fix: Use 'libgl1' instead of 'mesa-glx' for Python 3.11/Debian Bookworm
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app code
COPY model_functional_final.keras .
COPY app.py .

# Ensure logs are sent to Vertex AI immediately
ENV PYTHONUNBUFFERED=1

# Port 8080 is standard for Vertex AI
EXPOSE 8080

CMD ["python", "-u", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]