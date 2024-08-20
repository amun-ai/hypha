FROM continuumio/miniconda3

# Set working directory
WORKDIR /home

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

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
    libxrandr2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update conda and install dependencies
RUN conda update -n base -c defaults conda -y && \
    conda install pip -y

# Copy all files to the container
ADD . .

# Install Python dependencies
RUN pip install .[server-apps] --no-cache-dir

# Add user and switch to non-root user
RUN useradd -u 8877 hypha
USER hypha

# Install Playwright browsers as the hypha user
RUN playwright install --with-deps

# Expose port
EXPOSE 9520

# Define the command to run the application
CMD ["python", "-m", "hypha.server", "--host=0.0.0.0", "--port=9520"]
