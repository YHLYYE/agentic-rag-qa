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
        # 结构化抽取表格：把每个表格的行拼接成「单元格 | 单元格」文本
        table_lines: list[str] = []
        for table in page.find_tables().tables:
            for row in table.extract():
                cells = [str(c).strip() if c else "" for c in row]
                table_lines.append(" | ".join(cells))
        if table_lines:
            text = text + "\n[表格]\n" + "\n".join(table_lines)
        if text:
            pages.append(text)
    doc.close()
    return pages
