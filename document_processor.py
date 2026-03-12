import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts all text from a PDF file object.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_pages_from_pdf(pdf_file) -> list:
    """
    Extracts text per page to keep page numbers for citation.
    Returns a list of dicts: [{'page': 1, 'text': '...'}, ...]
    """
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append({"page": i + 1, "text": page_text})
    return pages

def get_document_chunks(pages, chunk_size=1000, chunk_overlap=200):
    """
    Splits the page texts into chunks while retaining page number metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for page_data in pages:
        page_chunks = text_splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {"page": page_data["page"]}
            })
    return chunks
