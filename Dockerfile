ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
FROM ${BASE_IMAGE}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7865 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# The nvidia/pytorch image already has python, pip, git, etc.
# We just need to ensure generic libs are present if missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
# Note: NVIDIA NGC containers often run as root by default, but good practice to use non-root if possible.
# However, for simplicity with NGC volumes, we'll stick to typical NGC patterns or ensure perms.
# Let's create an appuser.
# RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# COPY dependencies first to cache pip install
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hf_transfer && \
    pip install --no-cache-dir -r requirements.txt
# pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 # Removed specialized index as NGC image has torch

# Download spacy model if needed (based on requirements having spacy)
RUN python -m spacy download en_core_web_sm

# COPY the rest of the application
COPY . /app

# Install the application
RUN pip install .

# Ensure target directories for volumes exist and have correct initial ownership
RUN mkdir -p /app/outputs /app/checkpoints /app/logs
# chown -R appuser:appuser /app/outputs /app/checkpoints /app/logs /app

# Switch to non-root user
# USER appuser

# Expose the port the app runs on
EXPOSE 7865

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Set healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=5 \
    CMD curl -f http://localhost:7865/ || exit 1

# Command to run the application with GPU support
CMD ["python3", "acestep/gui.py", "--server_name", "0.0.0.0", "--bf16", "true"]
