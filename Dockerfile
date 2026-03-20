FROM vision-deepstream:8.0-python

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Project layout
WORKDIR /app
COPY bpa_vision/ /app/bpa_vision/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY deepstream/ /app/deepstream/

ENV PYTHONPATH=/app:/app/deepstream
