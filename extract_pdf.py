import fitz  # PyMuPDF
import re


# ================= EXTRACT =================

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        print(f"✅ Extracted {len(text)} characters from PDF")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
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


# ================= SAFE CHUNK SIZE =================
# Gemini embedding-001 has a token limit (~2048 tokens).
# Using 200 words per chunk to stay safely within limits.
# Overlap of 40 words preserves context across boundaries.

CHUNK_SIZE = 200   # words per chunk — safe for Gemini embedding
OVERLAP = 40       # word overlap between chunks


def split_into_word_chunks(text, prefix=""):
    """Split a block of text into word-limited chunks with overlap."""
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
    """
    Section-aware chunking:
    1. Detects headings and groups text by section.
    2. Tags each chunk with [SECTION HEADING] for better retrieval context.
    3. Falls back to plain sliding window if no sections found.
    4. Keeps chunk sizes within Gemini embedding token limits.
    """
    lines = text.split('\n')
    sections = []
    current_heading = "General"
    current_body = []

    for line in lines:
        if is_heading(line):
            if current_body:
                sections.append((current_heading, " ".join(current_body)))
            current_heading = line.strip()
            current_body = []
        else:
            stripped = line.strip()
            if stripped:
                current_body.append(stripped)

    # Flush last section
    if current_body:
        sections.append((current_heading, " ".join(current_body)))

    # If we detected real sections, use section-aware chunking
    if len(sections) > 3:
        all_chunks = []
        for heading, body in sections:
            prefix = f"[{heading}] "
            section_chunks = split_into_word_chunks(body, prefix=prefix)
            all_chunks.extend(section_chunks)
        print(f"✅ Section-aware: {len(sections)} sections → {len(all_chunks)} chunks")
        return all_chunks

    # Fallback: plain sliding window
    all_chunks = split_into_word_chunks(text)
    print(f"✅ Fallback sliding window: {len(all_chunks)} chunks")
    return all_chunks