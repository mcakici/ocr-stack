FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-tur \
    poppler-utils qpdf ghostscript \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pillow ocrmypdf beautifulsoup4

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 9880
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9880", "--workers", "6"]
