
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg libavcodec-extra libdav1d7 && \
    rm -rf /var/lib/apt/lists/*

COPY app/app.py .
COPY app/start.sh .
RUN chmod +x start.sh


COPY ../pip_depends/ ./wheels/


RUN pip install --no-cache-dir --no-index --find-links ./wheels ./wheels/*.whl

EXPOSE 8117
CMD ["/app/start.sh"]