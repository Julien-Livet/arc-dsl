FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN [ -e ARC-AGI-2 ] || git clone https://github.com/arcprize/ARC-AGI-2.git
RUN mkdir scripts
WORKDIR /app/scripts
COPY . arc-dsl
WORKDIR /app/scripts/arc-dsl
RUN python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
