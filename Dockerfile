FROM nvcr.io/nvidia/deepstream:8.0-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python dependencies
RUN pip3 install --no-cache-dir \
    pydantic>=2.0 \
    pyyaml>=6.0

# Project layout
WORKDIR /app
COPY bpa_vision/ /app/bpa_vision/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY deepstream/ /app/deepstream/

ENV PYTHONPATH=/app
