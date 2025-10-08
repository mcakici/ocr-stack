from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import subprocess, tempfile, os, shlex

app = FastAPI(title="OCR Service", version="1.0")

@app.get("/")
def health():
    return {"ok": True}

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form("tur"),      # Türkçe
    psm: int = Form(6),           # sayfa segmentasyonu
    oem: int = Form(1),           # LSTM engine
):
    # Görüntü dosyasını diske yaz
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # Tesseract çıktısını stdout'a yazdır
        cmd = ["tesseract", tmp_path, "stdout", "-l", lang, "--oem", str(oem), "--psm", str(psm)]
        proc = run(cmd)
        if proc.returncode != 0 and not proc.stdout.strip():
            return JSONResponse({"error": proc.stderr.strip()}, status_code=500)
        return {"text": proc.stdout}
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/ocr-pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    lang: str = Form("tur"),
    force_ocr: bool = Form(False)   # True ise her koşulda OCR yapar
):
    # PDF'i geçici dosyaya yaz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        data = await file.read()
        tmp_pdf.write(data)
        pdf_path = tmp_pdf.name

    out_txt = tempfile.mktemp(suffix=".txt")
    out_pdf = tempfile.mktemp(suffix=".pdf")

    try:
        # 1) Önce PDF'in metni var mı diye hızlıca dene (gömülü metin)
        if not force_ocr:
            pdftotext = run(["pdftotext", "-layout", pdf_path, "-"])  # stdout
            text_quick = pdftotext.stdout.strip()
            if text_quick:
                return {"text": text_quick, "source": "embedded-text"}

        # 2) Gömülü metin yoksa veya force_ocr ise OCRmyPDF ile OCR yap + sidecar text üret
        cmd = ["ocrmypdf", "--language", lang, "--sidecar", out_txt, "--optimize", "0"]
        
        # force_ocr parametresine göre davranış belirle
        if force_ocr:
            # Tüm sayfaları OCR'le (embedded text varsa bile üzerine yaz)
            cmd.append("--force-ocr")
        else:
            # Sadece metin olmayan sayfaları OCR'le
            cmd.append("--skip-text")
        
        # PDF dosyalarını ekle
        cmd.extend([pdf_path, out_pdf])
        
        proc = run(cmd)
        
        # OCRmyPDF başarılı olsa bile stderr'a bilgi mesajları yazabilir
        # Sadece return code 0 değilse hata döndür
        if proc.returncode != 0:
            # Ama önce çıktı dosyası oluşmuş mu kontrol et
            if not os.path.exists(out_txt) or os.path.getsize(out_txt) == 0:
                return JSONResponse({"error": f"OCR failed: {proc.stderr.strip()}"}, status_code=500)
        
        # sidecar'ı oku
        text = ""
        if os.path.exists(out_txt):
            with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        
        # Eğer text boşsa ve hata varsa hata döndür
        if not text.strip() and proc.returncode != 0:
            return JSONResponse({"error": f"No text extracted: {proc.stderr.strip()}"}, status_code=500)
            
        return {"text": text, "source": "ocr"}
    finally:
        for p in [pdf_path, out_txt, out_pdf]:
            try: os.remove(p)
            except: pass
