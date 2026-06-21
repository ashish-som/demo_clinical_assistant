# pdf_loader.py

'''
PDF Loading Utilities
Extracts text from PDFs using PyMUPDF Python module
'''

from pathlib import Path
from typing import Iterable, List
import fitz
from langchain_core.documents import Document
# print('Import Successful')

def extract_study_metadata(file_path: Path) -> dict:
    '''Creates a simple metadata from the PDF file name'''

    file_name = file_path.stem
    parts = file_name.split('_', maxsplit=1)

    return {
        'study_id': parts[0] if parts else 'unknown',
        'author_year': parts[1] if len(parts)>1 else file_name,
        'source_file': file_path.name
    }


def load_pdf(file_path:Path) -> List[Document]:
    '''Reads one PDF and returns one Document per non-empty page'''

    metadata = extract_study_metadata(file_path=file_path)
    documents: List[Document] =[]

    with fitz.open(file_path) as pdf:
        for page_index, page in enumerate(pdf):
            text=page.get_text().strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        'page_number': page_index + 1
                    }
                )
            )
    return documents


def load_pdfs(pdf_paths: Iterable[Path]) -> List[Document]:
    '''Reads many PDFs and combines their page Documents'''

    all_documents: List[Document] = []

    for file_path in pdf_paths:
        if file_path.suffix.lower() == '.pdf':
            all_documents.extend(load_pdf(file_path=file_path))

    return all_documents


def list_pdf_files(pdf_dir:Path) -> List[Path]:
    'Returns all PDFs in a folder in a sorted order'

    if not pdf_dir.exists():
        return []
    return sorted(pdf_dir.glob("*.pdf"))

        



