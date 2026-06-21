# services.py

'''
Combines services for lower level functions into a real production workflow
'''

from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.agents import(
    build_agentic_rag_graph,
    build_paper_summary_graph
)

from src.chunking import chunk_documents
from src.config import(
    CHAT_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    VECTOR_DB_DIR
)

from src.interfaces import ChunkingStrategy, DocumentLoader
from src.pdf_loader import load_pdfs
from src.retriever import(
    get_all_vectorstore_documents,
    hybrid_retrieve_with_query_variations
)
from src.vector_store import( 
    build_vector_store, 
    load_vector_store,
    vector_store_exists
)

@dataclass
class IndexingResult:
    'Summary of what happend during PDF indexing'

    pages: int
    chunks: int
    vectorstore: object

class PdfDocumentLoader:
    
    '''
    loads PDFs into page level Documents
    '''

    def load(self, paths: list[Path]):
        """Loads PDFs from the disk"""

        return load_pdfs(paths)


class SectionChunker:
    
    """Chunks pages and labels each chunk with metadata"""

    def __init__(
            self,
            chunk_size = DEFAULT_CHUNK_SIZE,
            chunk_overlap = DEFAULT_CHUNK_OVERLAP
            ):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        'Splits page Documents into section aware chunks'

        return chunk_documents(
            documents=documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )    


class PaperIndexingService:
    
    '''
    Runs the complete ingestion pipeline for PDFs to Vector Databases
    '''

    def __init__(
            self,
            loader: DocumentLoader | None =  None,
            chunker: ChunkingStrategy | None = None,
            persist_dir: Path = VECTOR_DB_DIR
                 ):
        
        self.loader = loader or PdfDocumentLoader()
        self.chunker = chunker or SectionChunker()
        self.persist_dir = persist_dir


    
    def index(self, pdf_paths: list[Path]) -> IndexingResult:

        '''
        Loads PDFs, create chunks, embed chunks, persists vector store
        '''

        pages = self.loader.load(pdf_paths)
        chunks = self.chunker.split(pages)
        vectorstore = build_vector_store(chunks=chunks,
                                         persist_dir=self.persist_dir)
        
        return IndexingResult(
            pages = len(pages),
            chunks = len(chunks),
            vectorstore = vectorstore
        )

class HybridResearchRetriever:

    """
    Retrives evidence using query variation. vector search and keyword search
    """

    def __init__(self, vectorstore, top_k: int = DEFAULT_TOP_K, llm=None):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.llm = llm or ChatOpenAI(model = CHAT_MODEL, temperature = 0)


    def retrieve(self, query: str):
        """Return relevant chunks for a research paper"""

        return hybrid_retrieve_with_query_variations(
            vectorstore=self.vectorstore,
            query=query,
            llm = self.llm,
            top_k = self.top_k
        )
    

class ClinicalResearchAssistant:
    """High level assisatnt used by the streamlit app and future API"""

    def __init__(self, vectorstore,
                    top_k:int = DEFAULT_TOP_K):
        self.vectorstore = vectorstore
        self.top_k = top_k


    def ask(self, question: str) -> dict:
        """Answers a research question with cited evidence"""

        graph = build_agentic_rag_graph(vectorstore = self.vectorstore,
                                        top_k = self.top_k)
        
        return graph.invoke({"query": question})
    

    def summarise(self, focus: str = "Summarise the indexed research papers") -> dict:
        """Generates a structured summary across the index papers"""

        graph = build_paper_summary_graph(
            vectorstore = self.vectorstore,
            top_k = self.top_k
        )

        return graph.invoke({'query': focus})
    

    def list_papers(self) -> list[str]:
        """Return unique paper names available in the vector database"""

        docs = get_all_vectorstore_documents(vectorstore = self.vectorstore)
        papers = {doc.metadata.get('source_file', 'unknown') for doc in docs}
        return sorted(papers)
    

def load_existing_vectorstore(persist_dir:Path = VECTOR_DB_DIR):
    """Loads the persisted vector store if it already exists"""

    if vector_store_exists(persist_dir= persist_dir):
        return load_vector_store(persist_dir = persist_dir)
    
    return None
