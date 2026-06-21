# interfaces.py

'''
Interfaces with required components
'''

from pathlib import Path
from typing import Protocol
from langchain_core.documents import Document

class DocumentLoader(Protocol):
    'A component that converts files into langchain Documents.'

    def load(self, paths: list[Path]) -> list[Document]:
        '''Loads source files and returns page-level documents'''



class ChunkingStrategy(Protocol):
    'A component that splits Docuemnts into searchable chunks'

    def split(self, documents: list[Document]) -> list[Document]:
        '''Splits documents and attaches retrieval metadata'''



class RetrievalStrategy(Protocol):
    'A component that retrieves evidence for a given query'

    def retrieve(self, query: str) -> list[Document]:
        '''Returns relevant chunks for a given query'''