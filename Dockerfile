FROM mambaorg/micromamba:latest

# WORKDIR /home
# Copy your environment.yml into the image
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
RUN micromamba install -y -n base --file env.yaml
RUN micromamba clean --all --yes 

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

USER root
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
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

USER $MAMBA_USER 
EXPOSE 9527
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m","hypha.server" ]
