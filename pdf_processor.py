import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text +=content+'\n'
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    return splitter.split_text(text)