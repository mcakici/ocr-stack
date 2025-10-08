from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import subprocess, tempfile, os, shlex
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

app = FastAPI(title="OCR Service", version="1.0")
MAX_WORKERS = 4

@app.get("/")
def health():
    return {"ok": True}

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)

def ocr_single_page(png_file: str, lang: str) -> str:
    cmd = ["tesseract", png_file, "stdout", "-l", lang, "--oem", "1", "--psm", "3"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    if proc.returncode == 0:
        return proc.stdout.strip()
    return ""


@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = Form("tur"),
    psm: int = Form(6),
    oem: int = Form(1),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        cmd = ["tesseract", tmp_path, "stdout", "-l", lang, "--oem", str(oem), "--psm", str(psm)]
        proc = run(cmd)
        if proc.returncode != 0 and not proc.stdout.strip():
            return JSONResponse({"error": proc.stderr.strip()}, status_code=500)
        return {"text": proc.stdout}
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/ocr-hocr")
async def ocr_hocr(
    file: UploadFile = File(...),
    lang: str = Form("tur"),
    psm: int = Form(3),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name

    try:
        output_base = tempfile.mktemp()
        cmd = ["tesseract", tmp_path, output_base, "-l", lang, "--psm", str(psm), "hocr"]
        proc = run(cmd)
        
        hocr_file = output_base + ".hocr"
        if os.path.exists(hocr_file):
            with open(hocr_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            try: os.remove(hocr_file)
            except: pass
            
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=html_content)
        else:
            return JSONResponse({"error": "hOCR generation failed", "stderr": proc.stderr}, status_code=500)
    finally:
        try: os.remove(tmp_path)
        except: pass

@app.post("/ocr-pdf")
async def ocr_pdf(
    file: UploadFile = File(...),
    lang: str = Form("tur"),
    force_ocr: bool = Form(False),
    dpi: int = Form(200)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        data = await file.read()
        tmp_pdf.write(data)
        pdf_path = tmp_pdf.name

    try:
        if not force_ocr:
            pdftotext = run(["pdftotext", "-layout", pdf_path, "-"])
            text_quick = pdftotext.stdout.strip()
            if text_quick and len(text_quick) > 100:
                return {"text": text_quick, "source": "embedded-text"}

        png_dir = tempfile.mkdtemp()
        try:
            proc = run(["pdfimages", "-png", "-r", str(dpi), pdf_path, f"{png_dir}/page"])
            
            if proc.returncode != 0:
                proc = run(["pdftoppm", "-png", "-r", str(dpi), pdf_path, f"{png_dir}/page"])
                
            if proc.returncode != 0:
                return JSONResponse({"error": f"PDF to image conversion failed: {proc.stderr.strip()}"}, status_code=500)
            
            png_files = []
            i = 1
            while True:
                possible_names = [
                    f"{png_dir}/page-{i:03d}.png",
                    f"{png_dir}/page-{i:02d}.png",
                    f"{png_dir}/page-{i}.png",
                    f"{png_dir}/page{i:03d}.png",
                    f"{png_dir}/page{i:02d}.png",
                    f"{png_dir}/page{i}.png",
                ]
                found = False
                for png_file in possible_names:
                    if os.path.exists(png_file):
                        png_files.append(png_file)
                        found = True
                        break
                if not found:
                    break
                i += 1
            
            if not png_files:
                return JSONResponse({"error": "No images extracted from PDF"}, status_code=500)
            
            from functools import partial
            ocr_func = partial(ocr_single_page, lang=lang)
            
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(executor.map(ocr_func, png_files))
            
            all_text = [text for text in results if text]
            
            if not all_text:
                return JSONResponse({"error": "No text extracted from any page"}, status_code=500)
            
            final_text = "\n\n".join(all_text)
            return {"text": final_text, "source": "ocr", "pages": len(png_files)}
            
        finally:
            try:
                import shutil
                shutil.rmtree(png_dir)
            except:
                pass
                
    finally:
        try: os.remove(pdf_path)
        except: pass

@app.post("/ocr-pdf-hocr")
async def ocr_pdf_hocr(
    file: UploadFile = File(...),
    lang: str = Form("tur"),
    dpi: int = Form(200)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        data = await file.read()
        tmp_pdf.write(data)
        pdf_path = tmp_pdf.name

    try:
        import re
        pdf_info = run(["pdfinfo", pdf_path])
        pdf_title = "document"
        if pdf_info.returncode == 0:
            for line in pdf_info.stdout.split('\n'):
                if line.startswith('Title:'):
                    title = line.split('Title:')[1].strip()
                    if title and title != 'Untitled':
                        pdf_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:50]
                    break
        
        if not pdf_title or pdf_title == "document":
            pdf_title = os.path.splitext(file.filename or "document")[0]
            pdf_title = re.sub(r'[^\w\s-]', '', pdf_title).strip().replace(' ', '_')[:50]
        
        png_dir = tempfile.mkdtemp()
        try:
            proc = run(["pdftoppm", "-png", "-r", str(dpi), pdf_path, f"{png_dir}/{pdf_title}"])
            
            if proc.returncode != 0:
                return JSONResponse({"error": f"PDF to image conversion failed: {proc.stderr.strip()}"}, status_code=500)
            
            import glob
            png_files = sorted(glob.glob(f"{png_dir}/*.png"))
            
            if not png_files:
                return JSONResponse({"error": "No images extracted from PDF"}, status_code=500)
            
            all_hocr = []
            all_hocr.append('<?xml version="1.0" encoding="UTF-8"?>')
            all_hocr.append('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')
            all_hocr.append('<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="tr" lang="tr">')
            all_hocr.append('<head>')
            all_hocr.append('<title></title>')
            all_hocr.append('<meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>')
            all_hocr.append('<meta name="ocr-system" content="tesseract 5.5.0" />')
            all_hocr.append('<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_dir ocrp_lang ocrp_wconf"/>')
            all_hocr.append('</head>')
            all_hocr.append('<body>')
            
            for idx, png_file in enumerate(png_files, 1):
                output_base = tempfile.mktemp()
                cmd = ["tesseract", png_file, output_base, "-l", lang, "--psm", "3", "hocr"]
                proc = run(cmd)
                
                hocr_file = output_base + ".hocr"
                if os.path.exists(hocr_file):
                    with open(hocr_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    page_div = soup.find('div', class_='ocr_page')
                    
                    if page_div:
                        page_div['id'] = f'page_{idx}'
                        all_hocr.append(str(page_div))
                    
                    try: os.remove(hocr_file)
                    except: pass
            
            all_hocr.append('</body>')
            all_hocr.append('</html>')
            
            html_content = '\n'.join(all_hocr)
            
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=html_content)
                
        finally:
            try:
                import shutil
                shutil.rmtree(png_dir)
            except:
                pass
                
    finally:
        try: os.remove(pdf_path)
        except: pass

