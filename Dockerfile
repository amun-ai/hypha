FROM continuumio/miniconda3
WORKDIR /home
RUN mkdir /home/bin && \
    cd /home/bin && wget https://dl.min.io/server/minio/release/linux-amd64/minio && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod -R 777 /home/bin
RUN mkdir /.mc && \
    chmod -R 777 /.mc
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation\
    libasound2\
    libatk-bridge2.0-0\
    libatk1.0-0\
    libatspi2.0-0\
    libcairo2\
    libcups2\
    libdbus-1-3\
    libdrm2\
    libgbm1\
    libglib2.0-0\
    libgtk-3-0\
    libnspr4\
    libnss3\
    libpango-1.0-0\
    libx11-6\
    libxcb1\
    libxcomposite1\
    libxdamage1\
    libxext6\
    libxfixes3\
    libxrandr2
RUN conda update pip -y
ADD . .
RUN pip install .[server-apps]
# RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir playwright && playwright install
EXPOSE 3000
