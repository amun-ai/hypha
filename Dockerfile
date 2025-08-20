# Use official Python base image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /home

# Install system dependencies and tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    # Dependencies for Playwright
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libx11-xcb1 \
    libxss1 \
    libxtst6 \
    libxshmfence1 \
    ca-certificates \
    libcurl4 \
    libx11-xcb-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install MinIO server and client
RUN mkdir -p /home/bin && \
    cd /home/bin && \
    wget https://dl.min.io/server/minio/release/linux-amd64/minio && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x /home/bin/minio /home/bin/mc && \
    chmod -R 777 /home

# Create and set permissions for .mc directory
RUN mkdir -p /.mc && \
    chmod -R 777 /.mc

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements and setup files first for better caching
COPY requirements.txt setup.py ./
COPY hypha/VERSION ./hypha/VERSION

# Create virtual environment and install dependencies using uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Install Python dependencies with uv
RUN uv pip install -r requirements.txt && \
    uv pip install redis==5.2.0 \
    aiobotocore>=2.1.0 \
    aiortc>=1.9.0 \
    requests>=2.26.0 \
    playwright>=1.51.0 \
    base58>=2.1.0 \
    pymultihash>=0.8.2

# Copy all application files
COPY . .

# Install the package itself
RUN uv pip install -e .

# Install and setup Playwright
RUN playwright install --with-deps

# Add user and set permissions
RUN useradd -u 8877 hypha && \
    mkdir -p /.cache && \
    mv /root/.cache/ms-playwright /.cache && \
    chown -R hypha:hypha /.cache && \
    chown -R hypha:hypha /opt/venv

# Switch to non-root user
USER hypha

# Expose port
EXPOSE 9520

# Define the command to run the application
CMD ["python", "-m", "hypha.server", "--host=0.0.0.0", "--port=9520"]