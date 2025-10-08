# Küçük ve hızlı bir taban: Debian (slim)
FROM python:3.11-slim

# Sistem bağımlılıkları: tesseract, Türkçe dil paketi, pdf araçları, gs, qpdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-tur \
    poppler-utils qpdf ghostscript \
 && rm -rf /var/lib/apt/lists/*

# Python paketleri: API ve OCRmyPDF (Python arayüzü)
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pillow ocrmypdf

# (İsteğe bağlı) Tesseract tessdata yolu; çoğu kurulumda gerekmez.
# ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 9880
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9880"]
