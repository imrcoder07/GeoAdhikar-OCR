# app.py
"""
DB-FREE PDF-only OCR web app with Gemini post-processing + PDF export.

- Accepts only PDF uploads.
- Hybrid text extraction:
    1) Try pdfminer (digital PDFs) -> if good text, use it and skip OCR.
    2) Else convert PDF -> images (pdf2image) and OCR with Tesseract (scanned PDFs).
- OCR uses word-confidence filtering to drop garbage.
- Optional Gemini post-processing with custom prompt per upload.
- Saves JSON to data/json/<timestamp>_<filename>.json
- Exports a clean PDF report of results.
"""

import os
import re
import io
import json
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from loguru import logger

from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2

# Set OpenCV for headless mode
cv2.setUseOptimized(False)
os.environ['DISPLAY'] = ''

from app.core.field_extractor import FieldExtractor
from app.utils.normalization import (
    normalize_name, parse_date, parse_area, normalize_state, normalize_claim_type, clean_patta_id
)
from config.config import Config

# ---------- Optional libs ----------
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_OK = True
except Exception:
    PDFMINER_OK = False

try:
    from langdetect import detect
    LANGDETECT_OK = True
except Exception:
    LANGDETECT_OK = False

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent

# Tesseract config for Docker/Linux - use 'tesseract' to rely on PATH
if 'TESSDATA_PREFIX' not in os.environ:
    os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4/tessdata"
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Verify Tesseract availability
try:
    version = pytesseract.get_tesseract_version()
    logger.info("Tesseract version: %s", version)
except Exception as e:
    logger.warning("Tesseract executable not found or failed to initialize: %s. OCR will fail.", e)
    pytesseract.pytesseract.tesseract_cmd = None

# ---------- Gemini ----------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

UPLOAD_FOLDER = BASE_DIR / "app" / "static" / "uploads"
JSON_FOLDER = BASE_DIR / "data" / "json"
EXPORT_FOLDER = BASE_DIR / "data" / "exports"
for d in (UPLOAD_FOLDER, JSON_FOLDER, EXPORT_FOLDER, BASE_DIR / "logs"):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

# ---------- Logging ----------
logger.remove()
logger.add(BASE_DIR / "logs" / "app_no_db_gemini.log", rotation="10 MB", level="DEBUG")
logger.info("App start | tesseract=system | TESSDATA_PREFIX=%s | pdfminer=%s | langdetect=%s",
            os.environ.get("TESSDATA_PREFIX"), PDFMINER_OK, LANGDETECT_OK)

# ---------- Flask ----------
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Error handlers to ensure JSON responses
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

# ---------- Regex helpers ----------
# (Deprecated: Now using FieldExtractor)

# ---------- State ----------
ocr_processor = None

def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    pil_pages = convert_from_path(pdf_path, dpi=dpi, thread_count=1)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]

def normalize_lang_list(lang_input: str) -> str:
    if not lang_input:
        return "auto"
    token = lang_input.strip().lower()
    if token == "auto":
        return "auto"
    parts = [p.strip().lower() for p in lang_input.replace("+", ",").split(",") if p.strip()]
    mapping = {"hi": "hin", "hin": "hin", "en": "eng", "eng": "eng", "odia": "ori", "ori": "ori", "telugu": "tel", "tel": "tel"}
    out, seen = [], set()
    for p in [mapping.get(x, x) for x in parts]:
        if p not in seen:
            out.append(p); seen.add(p)
    return "+".join(out) if out else "auto"

def ocr_image_conf_filtered(img_bgr: np.ndarray, langs: str) -> Tuple[str, float]:
    """OCR with confidence filtering to reduce garbage text."""
    if pytesseract.pytesseract.tesseract_cmd is None:
        logger.warning("Tesseract not available; skipping OCR.")
        return "", 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Use data API to get per-word conf
    data = pytesseract.image_to_data(rgb, lang=langs, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
    words, confs = data.get("text", []), data.get("conf", [])
    kept, confs_kept = [], []
    for w, c in zip(words, confs):
        try:
            ci = int(c)
        except Exception:
            continue
        w = (w or "").strip()
        if ci > 30 and w:
            kept.append(w)
            confs_kept.append(ci)
    clean_text = " ".join(kept).strip()
    avg_conf = (sum(confs_kept) / len(confs_kept) / 100.0) if confs_kept else 0.0
    return clean_text, avg_conf

def extract_regex_fields(text: str) -> Dict[str, Any]:
    extractor = FieldExtractor()
    raw_fields = extractor.extract(text)
    # Flatten for compatibility
    fields = {}
    for key, data in raw_fields.items():
        fields[key] = {
            "value": data.get("value"),
            "raw": data.get("raw"),
            "confidence": data.get("confidence"),
            "note": data.get("note"),
            "source": data.get("source")
        }
    return fields

def normalize_gemini_fields(gemini_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Apply normalization to Gemini-extracted fields where applicable."""
    normalized = {}
    for key, val in gemini_raw.items():
        if key == "holder_name" and isinstance(val, str):
            norm_val, notes = normalize_name(val)
            normalized[key] = {"value": norm_val, "normalized": True, "notes": notes}
        elif key == "issue_date" and isinstance(val, str):
            iso_date, note = parse_date(val)
            normalized[key] = {"value": iso_date, "normalized": True, "note": note}
        elif key == "area_hectares" and isinstance(val, (str, float)):
            # Assume string or number input
            area_str = str(val) if isinstance(val, (int, float)) else val
            area_ha, note = parse_area(area_str)
            normalized[key] = {"value": area_ha, "normalized": True, "note": note}
        elif key == "state" and isinstance(val, str):
            norm_state = normalize_state(val)
            normalized[key] = {"value": norm_state, "normalized": True}
        elif key == "claim_type" and isinstance(val, str):
            norm_claim = normalize_claim_type(val)
            normalized[key] = {"value": norm_claim, "normalized": True}
        elif key == "patta_id" and isinstance(val, str):
            clean_id = clean_patta_id(val)
            normalized[key] = {"value": clean_id, "normalized": True}
        else:
            normalized[key] = {"value": val, "normalized": False}
    return normalized

# ---------- Gemini ----------
def init_ocr_processor():
    global ocr_processor
    ocr_processor = None
    if not GEMINI_AVAILABLE:
        logger.warning("google-generativeai not available; Gemini disabled.")
        return
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; Gemini disabled.")
        return
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini configured.")
        ocr_processor = True
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")

def gemini_extract_fields(raw_text: str, prompt_override: str | None, model_name: str = "gemini-1.5-flash") -> Tuple[Dict[str, Any], str]:
    if not GEMINI_AVAILABLE or not ocr_processor:
        return {}, "disabled"
    base_prompt = """
You are a data extraction assistant for Indian land/FRA documents.
Extract as JSON these fields with values and (if possible) confidence 0–1:
patta_id, holder_name, relative_name, relation_type, village, district, state, area_hectares, claim_type, claim_status, issue_date.
If a field is missing, set it to null. Return JSON ONLY, no extra text.
"""
    final_prompt = (prompt_override or "").strip() or base_prompt
    try:
        model = genai.GenerativeModel(model_name)
        full_prompt = f"{final_prompt}\n\nTEXT:\n'''{raw_text}'''"
        resp = model.generate_content(full_prompt)
        resp_text = resp.text
        s, e = resp_text.find("{"), resp_text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(resp_text[s:e+1]), "success"
            except Exception as je:
                logger.error("Gemini parse failed: %s | raw: %s", je, resp_text[:400])
                return {}, "error"
        logger.error("Gemini response lacked JSON. Raw: %s", resp_text[:400])
        return {}, "error"
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return {}, "error"

init_ocr_processor()

# ---------- pdfminer helper ----------
def extract_text_pdfminer(pdf_path: str) -> str:
    if not PDFMINER_OK:
        return ""
    try:
        txt = pdfminer_extract_text(pdf_path) or ""
        return re.sub(r"[ \t]+", " ", txt).strip()
    except Exception as e:
        logger.warning("pdfminer failed: %s", e)
        return ""

# ---------- PDF report (ReportLab) ----------
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import textwrap
from unidecode import unidecode

def build_pdf_report(doc_json: dict, out_path: Path):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    doc = SimpleDocTemplate(str(out_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, spaceAfter=20)
    story.append(Paragraph("GeoAdhikar – OCR Report", title_style))

    # Metadata
    meta_data = [
        ["Document ID", doc_json.get("document_id")],
        ["Original Filename", doc_json.get("original_filename")],
        ["Processed At", doc_json.get("processing_started_at")],
        ["Text Source", doc_json.get("text_source")],
        ["OCR Langs Used", doc_json.get("ocr_langs_used")],
        ["Dominant Language", doc_json.get("dominant_language")],
        ["OCR Confidence (avg)", doc_json.get("ocr_confidence_avg")],
        ["Gemini Status", doc_json.get("gemini_status")],
        ["Needs Review", doc_json.get("needs_review")],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 10*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Regex Fields Table
    story.append(Paragraph("Extracted Fields (Regex):", styles['Heading2']))
    regex_fields = doc_json.get("regex_fields") or {}
    if regex_fields:
        regex_data = [["Field", "Value", "Confidence", "Note"]]
        for k, v in regex_fields.items():
            if isinstance(v, dict):
                val = v.get("value", "")
                conf = v.get("confidence", "")
                note = v.get("note", "")
            else:
                val = v
                conf = ""
                note = ""
            regex_data.append([k, str(val), str(conf), str(note)])
        regex_table = Table(regex_data, colWidths=[3*cm, 5*cm, 2*cm, 5*cm])
        regex_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(regex_table)
    else:
        story.append(Paragraph("(none)", styles['Normal']))
    story.append(Spacer(1, 12))

    # Gemini Fields Table
    story.append(Paragraph("Extracted Fields (Gemini):", styles['Heading2']))
    gemini_fields = doc_json.get("gemini_fields") or {}
    if gemini_fields:
        gemini_data = [["Field", "Value", "Normalized", "Note"]]
        for k, v in gemini_fields.items():
            if isinstance(v, dict):
                val = v.get("value", "")
                norm = v.get("normalized", "")
                note = v.get("note", "")
            else:
                val = v
                norm = ""
                note = ""
            gemini_data.append([k, str(val), str(norm), str(note)])
        gemini_table = Table(gemini_data, colWidths=[3*cm, 5*cm, 2*cm, 5*cm])
        gemini_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(gemini_table)
    else:
        story.append(Paragraph("(none)", styles['Normal']))
    story.append(Spacer(1, 12))

    # OCR Text Preview (transliterated to avoid font issues)
    story.append(Paragraph("OCR Text Preview (Transliterated):", styles['Heading2']))
    raw_preview = (doc_json.get("preview") or "")[:4000]
    preview = unidecode(raw_preview)
    story.append(Paragraph(preview, styles['Normal']))

    doc.build(story)

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "ocr_ready": bool(ocr_processor),
        "pdfminer": PDFMINER_OK,
        "langdetect": LANGDETECT_OK
    })

@app.route("/api/upload", methods=["POST"])
def upload_file():
    start_ts = datetime.now(UTC)
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        user_prompt = request.form.get("prompt")  # optional Gemini prompt
        langs_param = request.form.get("languages", "auto")

        if not allowed_file(file.filename):
            return jsonify({"error": "PDF only"}), 400

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        logger.info("Saved uploaded file: %s", filepath)

        # 1) Digital text first
        all_text = ""
        text_source = "unknown"
        digital_text = extract_text_pdfminer(str(filepath)) if PDFMINER_OK else ""
        if len(digital_text) >= 200:
            all_text = digital_text
            text_source = "pdfminer"
            logger.info("Using pdfminer text (%d chars).", len(all_text))

        # 2) OCR fallback
        pages = []
        page_results = []
        avg_conf_accum = []
        normalized = normalize_lang_list(langs_param)

        if text_source != "pdfminer":
            try:
                pages = pdf_to_images(str(filepath), dpi=400)
            except Exception as e:
                logger.exception("pdf_to_images failed")
                return jsonify({"error": "Failed to convert PDF to images", "detail": str(e)}), 500
            if not pages:
                return jsonify({"error": "PDF conversion produced no pages"}), 500

            ocr_langs = "hin+eng" if normalized == "auto" else normalized
            # Debug: save first page
            if pages:
                cv2.imwrite('debug_page.png', pages[0])
                logger.info("Saved debug_page.png for troubleshooting")
            for i, page in enumerate(pages, start=1):
                try:
                    text, avg_conf = ocr_image_conf_filtered(page, langs=ocr_langs)
                except Exception:
                    logger.exception("OCR failed on page %s", i)
                    text, avg_conf = "", 0.0
                avg_conf_accum.append(avg_conf)
                all_text += f"\n\n=== PAGE {i} ===\n{text}"
                page_results.append({"page": i, "avg_confidence": round(avg_conf, 3), "snippet": text[:800]})
            text_source = "tesseract"
        else:
            # cheap page count probe
            try:
                pages = pdf_to_images(str(filepath), dpi=50)
            except Exception:
                pages = []
            page_results.append({"page": 1, "avg_confidence": 1.0, "snippet": all_text[:800]})

        page_count = len(pages) if pages else 1

        # 3) Language detect
        dominant_lang = "unknown"
        if LANGDETECT_OK and all_text.strip():
            try:
                dominant_lang = detect(all_text)
            except Exception:
                pass

        # 4) Regex fields (improved)
        regex_fields = extract_regex_fields(all_text)

        # 5) Gemini post-process (optional)
        gemini_raw, gemini_status = gemini_extract_fields(all_text, user_prompt)
        # Normalize Gemini outputs
        gemini_fields = normalize_gemini_fields(gemini_raw) if gemini_status == "success" else {}

        # 6) Review flag
        ocr_conf_avg = (sum(avg_conf_accum) / len(avg_conf_accum)) if avg_conf_accum else (1.0 if text_source == "pdfminer" else 0.0)
        needs_review = False
        reasons = []
        if text_source == "tesseract" and ocr_conf_avg < 0.6:
            needs_review, reasons = True, ["Low OCR confidence"]
        if not all_text.strip():
            needs_review, reasons = True, reasons + ["Empty extracted text"]
        for k in ["deed_number", "plot_identifier", "area"]:
            if not (regex_fields.get(k) or {}).get("value"):
                needs_review = True
                reasons.append(f"Missing key field: {k}")
                break

        result = {
            "document_id": filename,
            "original_filename": file.filename,
            "upload_path": str(filepath),
            "page_count": page_count,
            "pages": page_results,
            "text_source": text_source,
            "dominant_language": dominant_lang,
            "ocr_langs_used": "hin+eng" if normalized == "auto" else normalized,
            "ocr_confidence_avg": round(ocr_conf_avg, 3),
            "regex_fields": regex_fields,
            "gemini_fields": gemini_fields,
            "gemini_status": gemini_status,   # success | disabled | error
            "needs_review": needs_review,
            "review_reasons": reasons,
            "preview": all_text[:4000],
            "processing_started_at": start_ts.isoformat()
        }

        out_file = JSON_FOLDER / (filename + ".json")
        with open(out_file, "w", encoding="utf8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        logger.info("Saved JSON: %s", out_file)

        return jsonify(result)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Upload failed: %s\n%s", e, tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.route("/api/documents")
def list_docs():
    items = []
    for p in sorted(JSON_FOLDER.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, "r", encoding="utf8") as f:
                doc = json.load(f)
            items.append({"id": doc.get("document_id"), "name": doc.get("original_filename")})
        except Exception:
            continue
    return jsonify({"success": True, "documents": items})

@app.route("/api/document/<doc_id>")
def get_doc(doc_id):
    p = JSON_FOLDER / (doc_id + ".json")
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    with open(p, "r", encoding="utf8") as f:
        doc = json.load(f)
    return jsonify({"success": True, "document": doc})

# ---- PDF Export ----
@app.route("/api/export/<doc_id>/pdf", methods=["GET"])
def export_pdf(doc_id):
    json_path = JSON_FOLDER / (doc_id if doc_id.endswith(".json") else f"{doc_id}.json")
    if not json_path.exists():
        return jsonify({"error": "Document JSON not found"}), 404
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out_pdf = EXPORT_FOLDER / (doc_id.replace(".json", "") + ".report.pdf")
    try:
        build_pdf_report(data, out_pdf)
        return jsonify({"success": True, "filename": out_pdf.name})
    except Exception as e:
        logger.exception("PDF export failed")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/download/<doc_id>.pdf", methods=["GET"])
def download_pdf(doc_id):
    pdf_path = EXPORT_FOLDER / f"{doc_id}.report.pdf"
    if not pdf_path.exists():
        return jsonify({"error": "Export not found. Call /api/export/<doc_id>/pdf first."}), 404
    return send_file(str(pdf_path), as_attachment=True, download_name=pdf_path.name, mimetype="application/pdf")

if __name__ == "__main__":
    logger.info("Starting OCR+Gemini app (DB-free, PDF only)")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
