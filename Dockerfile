FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 tesseract-ocr && rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /app

# copy project files
COPY . /app

# install python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]