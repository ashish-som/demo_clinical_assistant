# chunking.py

'''
Document Chunking Utility
Creates section aware chunks from documents as per the given chunk size and overlap
'''

import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# print('Import Successful')

def chunk_documents(
        documents: List[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    
    '''Split page Documents into smaller chunks for retrieval'''

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    chunks = add_section_labels(chunks)
    return add_chunk_ids(chunks)


def detect_section(text: str) -> str:
    '''Dectects a section heading inside a chunk'''

    first_lines = "\n".join(text.strip().splitlines()[:8]).lower()

    section_patterns = {
        'abstract': r"\babstract\b",
        'introduction': r'\bintroduction\b|\bbackground\b',
        'methods': r'\bmethods?\b|\bmaterials and methods\b',
        'results': r'\bresults?\b|\bfindings\b',
        'discussion': r'\bdiscussions?\b',
        'conclusion': r'\bconclusions?\b',
        'references': r'\breferences\b|\bbibliography\b'
    }

    for section, pattern in section_patterns.items():
        if re.search(pattern, first_lines):
            return section
        
    return 'unknown'

def add_section_labels(chunks: List[Document]) -> List[Document]:
    '''
    Attach a likely section name to a chunk
    '''

    current_section_by_file={}

    for chunk in chunks:
        source_file = chunk.metadata.get('source_file', 'unknown')
        detected_section = detect_section(text=chunk.page_content)

        if detected_section != 'unknown':
            current_section_by_file[source_file] = detected_section

        chunk.metadata['section'] = current_section_by_file.get(source_file, 'unknown')
        # chunk.metadata['section'] = detected_section

    return chunks


def add_chunk_ids(chunks: List[Document]) -> List[Document]:

    '''
    Attaches a unique ID to each chunk'''

    for index, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = index

    return chunks

