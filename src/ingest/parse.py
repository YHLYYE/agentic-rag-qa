import re
import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page in doc:
        text = clean_text(page.get_text())
        if text:
            pages.append(text)
    doc.close()
    return pages
