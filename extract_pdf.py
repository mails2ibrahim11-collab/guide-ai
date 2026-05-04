import fitz  # PyMuPDF
import re
import io
import pytesseract
from PIL import Image
from logger import get_logger

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

log = get_logger("extract_pdf")


# ================= OCR FALLBACK =================

def extract_text_from_page(page, page_num):
    text = page.get_text()
    if len(text.strip()) > 20:
        return text

    log.debug(f"[OCR] Page {page_num} is image-based — running OCR")
    try:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception:
        log.warning("[OCR] ⚠️ pytesseract not available — skipping OCR for this page")
        return ""


# ================= EXTRACT =================

def extract_text_from_pdf(file_path):
    log.info(f"[PDF] Opening '{file_path}'...")
    text = ""
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        log.debug(f"[PDF] Document has {total_pages} page(s)")

        for i, page in enumerate(doc):
            page_text = extract_text_from_page(page, i + 1)
            text += f"\n[Page {i + 1}]\n{page_text}\n"
            log.debug(f"[PDF] Page {i+1}/{total_pages} — {len(page_text)} chars extracted")

        log.info(f"[PDF] ✅ Extraction complete — {len(text)} total characters from {total_pages} page(s)")

    except Exception as e:
        log.error(f"[PDF] ❌ Failed to read '{file_path}': {e}")

    return text


# ================= SECTION DETECTION =================

SECTION_PATTERNS = [
    r'^\d+[\.\s]+[A-Z][^\n]{3,60}$',
    r'^[A-Z][A-Z\s]{4,40}$',
    r'^\s*(Chapter|Section|Part)\s+\d+',
]


def is_heading(line):
    line = line.strip()
    if not line or len(line) > 80:
        return False
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, line):
            return True
    if line.isupper() and 4 < len(line) < 60:
        return True
    return False


# ================= CHUNK SETTINGS =================

CHUNK_SIZE = 200
OVERLAP = 40


def split_into_word_chunks(text, prefix=""):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk = (prefix + " ".join(chunk_words)).strip()
        if chunk:
            chunks.append(chunk)
        i += (CHUNK_SIZE - OVERLAP)
    return chunks


# ================= SECTION-AWARE CHUNKING =================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    log.info(f"[CHUNK] Starting chunking ({len(text)} chars, chunk_size={chunk_size}, overlap={overlap})...")

    page_blocks = re.findall(r"\[Page\s+(\d+)\]\s*(.*?)(?=\n\[Page\s+\d+\]|\Z)", text, flags=re.S)
    if page_blocks:
        all_chunks = []
        for page_num, page_body in page_blocks:
            page_body = page_body.strip()
            if not page_body:
                continue
            page_chunks = split_into_word_chunks(page_body, prefix=f"[Page {page_num}] ")
            all_chunks.extend(page_chunks)
            log.debug(f"[CHUNK] Page {page_num} â†’ {len(page_chunks)} chunk(s)")

        if all_chunks:
            log.info(f"[CHUNK] âœ… Page-aware complete â€” {len(page_blocks)} page(s) â†’ {len(all_chunks)} chunks")
            return all_chunks

    lines = text.split('\n')
    log.debug(f"[CHUNK] Scanning {len(lines)} lines for section headings...")

    sections = []
    current_heading = "General"
    current_body = []
    headings_found = 0

    for line in lines:
        if is_heading(line):
            if current_body:
                sections.append((current_heading, " ".join(current_body)))
            current_heading = line.strip()
            current_body = []
            headings_found += 1
        else:
            stripped = line.strip()
            if stripped:
                current_body.append(stripped)

    if current_body:
        sections.append((current_heading, " ".join(current_body)))

    log.debug(f"[CHUNK] Detected {headings_found} heading(s) → {len(sections)} section(s)")

    # Section-aware path
    if len(sections) > 3:
        log.info(f"[CHUNK] Using section-aware chunking ({len(sections)} sections)")
        all_chunks = []
        for heading, body in sections:
            prefix = f"[{heading}] "
            section_chunks = split_into_word_chunks(body, prefix=prefix)
            all_chunks.extend(section_chunks)
            log.debug(f"[CHUNK] Section '{heading[:40]}' → {len(section_chunks)} chunk(s)")

        log.info(f"[CHUNK] ✅ Section-aware complete — {len(sections)} sections → {len(all_chunks)} chunks")
        return all_chunks

    # Fallback path
    log.warning(f"[CHUNK] ⚠️ Only {len(sections)} section(s) detected — falling back to sliding window")
    all_chunks = split_into_word_chunks(text)
    log.info(f"[CHUNK] ✅ Sliding window complete — {len(all_chunks)} chunks")
    return all_chunks
