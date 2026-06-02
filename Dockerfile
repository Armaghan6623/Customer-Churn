FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# JSON-array form handles the space in the folder name on all platforms
COPY ["customer crunch/requirements-hf.txt", "./requirements.txt"]
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace into the container
COPY . .

# Rename the folder with an underscore to prevent syntax and routing bugs
RUN mv "customer crunch" customer_crunch

# Create a non-privileged user for Hugging Face container compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

# Run the application from the newly renamed underscore directory
CMD ["python", "customer_crunch/ui/app.py"]