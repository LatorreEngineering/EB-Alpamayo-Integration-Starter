# Multi-stage Dockerfile for EB-Alpamayo-Integration-Starter
# Supports NVIDIA GPUs with CUDA 12.1+

# ==============================================================================
# Stage 1: Base image with CUDA and Python
# ==============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip and install build tools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ==============================================================================
# Stage 2: Development image with all dependencies
# ==============================================================================
FROM base AS development

WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml ./

# Install PyTorch with CUDA support first (explicit CUDA 12.1 version)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
RUN pip install --no-cache-dir -e .[dev]

# Install AlpaSim from GitHub (if available)
# Note: This may fail if the repo is not yet public; users can install manually
RUN pip install --no-cache-dir git+https://github.com/NVlabs/alpasim.git || \
    echo "AlpaSim not available - users must install manually"

# Install Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab-vim \
    jupyterlab-git \
    ipympl

# Set up Jupyter Lab
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Create non-root user for security best practices
RUN useradd -m -s /bin/bash alpamayo && \
    chown -R alpamayo:alpamayo /workspace

# Switch to non-root user
USER alpamayo

# Expose Jupyter Lab port
EXPOSE 8888

# Set default command to start Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]

# ==============================================================================
# Stage 3: Production-minimal image (optional, for deployment)
# ==============================================================================
FROM base AS production

WORKDIR /app

# Copy only necessary files
COPY pyproject.toml ./
COPY src/ ./src/

# Install production dependencies only
RUN pip install --no-cache-dir -e .

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Create non-root user
RUN useradd -m -s /bin/bash alpamayo && \
    chown -R alpamayo:alpamayo /app

USER alpamayo

# Default to Python interactive shell
CMD ["python"]

# ==============================================================================
# Build Instructions:
# ==============================================================================
# Development image (default):
#   docker build -t eb-alpamayo:dev --target development .
#   docker run --gpus all -p 8888:8888 -v $(pwd):/workspace eb-alpamayo:dev
#
# Production image:
#   docker build -t eb-alpamayo:prod --target production .
#   docker run --gpus all -v $(pwd)/data:/app/data eb-alpamayo:prod
#
# Testing GPU access:
#   docker run --gpus all eb-alpamayo:dev python -c "import torch; print(torch.cuda.is_available())"
# ==============================================================================
