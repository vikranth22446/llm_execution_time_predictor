FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace/

RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

CMD ["/bin/bash"]