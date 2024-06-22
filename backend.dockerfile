FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt
COPY backend.py /app/backend.py
COPY download_the_model.py /app/download_the_model.py

# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt
RUN pip install --no-cache-dir --upgrade -r requirements_backend.txt
# RUN python download_the_model.py

EXPOSE 8080
CMD exec uvicorn --port 8080 --host 0.0.0.0 backend:app