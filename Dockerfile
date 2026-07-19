FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TERM=xterm-256color \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# ttyd
RUN curl -L https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
    -o /usr/local/bin/ttyd \
    && chmod +x /usr/local/bin/ttyd

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install .

RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]