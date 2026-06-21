# retriever.py

'''Retrieves relevant chunks using different advanced retrieval techniques'''

import re
from collections import Counter
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from src.config import DEFAULT_TOP_K
# print('Import Successful')


def format_context(docs: List[Document]) -> str:
    '''Formats retrieved chunks into a citation friendly prompt'''

    context_blocks =[]

    for index, doc in enumerate(docs):
        context_blocks.append(
            f"""[Source: {index}]
Study: {doc.metadata.get('author_year'), 'unknown'}
File: {doc.metadata.get('source_file', 'unknown')}
Page: {doc.metadata.get('page_number', 'NA')}
Chunk ID: {doc.metadata.get('chunk_id', 'NA')}
Section: {doc.metadata.get('section','NA')}

Content:
{doc.page_content}
"""
        )

    return '\n-------------------------------\n'.join(context_blocks)


def create_query_variations(query: str, 
                            llm: ChatOpenAI, 
                            count:int=3) -> List[str]:
    '''
    Generate alternate paraphrasing of users' queries
    '''

    prompt=f'''
Create {count} search queries for retrieving evidence from clinical research papers.
Original Query:
{query}

Rules:
- Keep the meaning same.
- Use clinical/research terminologies whereever useful.
- Return one query per line
'''
    
    response = llm.invoke(prompt).content.strip()
    variations = [line.strip('-').strip() for line in response.splitlines()]
    variations = [line for line in variations if line]

    return [query] + variations[:count]


def tokenize(text: str) -> List[str]:
    '''
    Converts text into lower case
    '''

    return re.findall(r'[a-z0-9]+', text.lower())


def get_all_vectorstore_documents(vectorstore) -> List[Document]:

    '''
    Loads stored chunks from Chroma DB so keyworded search retrieval can be implemented
    '''

    raw = vectorstore.get(include = ['documents', 'metadatas'])
    documents = raw.get('documents', [])
    metadatas = raw.get('metadatas', [])

    return [
        Document(page_content=content, metadata = metadata or {})
        for content, metadata in zip(documents, metadatas)
    ]


def keyword_search(docs: List[Document], 
                   query: str, 
                   top_k: int = DEFAULT_TOP_K) -> List[Document]:
    '''
    Retrieves chunks by exact keyword overlap with the user query
    '''

    query_terms = Counter(tokenize(query))
    scored_docs=[]

    for doc in docs:
        doc_terms = Counter(tokenize(doc.page_content))
        score = sum(min(count, doc_terms.get(term, 0)) for term, count in query_terms.items())

        if score>0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item:item[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]


def infer_section_filter(query: str) -> str | None:
    
    '''
    Infers which section is the most useful for the given query
    '''

    query_lower= query.lower()

    section_keywords = {
        'methods': ['method', 'sample size', 'particpipants', 'cohort', 'design', 'criteria'],
        'results': ['result', 'outcome', 'findings', 'mortality', 'rate', 'proportion'],
        'discussion': ['discussion', 'interpretation', 'implication'],
        'conclusion': ['conclusion', 'conclude', 'summary'],
        'abstract': ['abstract', 'overview']
    }

    for section, keywords in section_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return section
    return None


def detect_requested_sources(query: str, docs: List[Document]) -> set[str]:
    
    '''
    Detects whether a query names a particular papaer or study
    '''

    query_lower = query.lower()
    requested_sources = set()

    for doc in docs:
        source_file = doc.metadata.get('source_file', '')
        author_year = doc.metadata.get('author_year', '')
        source_stem = source_file.lower().replace('.pdf','')

        candidates = [
            source_file.lower(),
            source_stem,
            author_year.lower()
        ]


        if any(candidate and candidate in query_lower for candidate in candidates):
            requested_sources.add(source_file)

    return requested_sources


def filter_by_requested_sources(docs: List[Document], requested_sources: set[str]) -> List[Document]:

    '''
    Keeps chunks only from the requested sources
    '''

    if not requested_sources:
        return docs
    
    return [doc for doc in docs if doc.metadata.get('source_file') in requested_sources]


def deduplicate_docs(docs: List[Document]) -> List[Document]:

    '''
    Removes duplictae/repeated chunks returned by multiple retrieval methods
    '''

    seen = set()
    unique_docs=[]

    for doc in docs:
        key =(
            doc.metadata.get('source_file'),
            doc.metadata.get('page_number'),
            doc.metadata.get('chunk_id')
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def diversify_by_sources(docs: List[Document], max_per_source: int = 3) -> List[Document]:

    '''
    Keeps evidence from multiple papers instead of one dominant paper
    '''

    source_counts = Counter()
    diverse_docs=[]

    for doc in docs:
        source = doc.metadata.get('source_file','Unknown')

        if source_counts[source] >= max_per_source:
            continue

        source_counts[source] += 1
        diverse_docs.append(doc)

    return diverse_docs


def retrieve_with_query_variations(
        retriever,
        query: str,
        llm: ChatOpenAI,
        variation_count: int = 3
) -> List[Document]:
    
    '''
    Retrieve chunks using main query plus generated variations
    '''

    seen = set()
    results: List[Document] = []

    for search_query in create_query_variations(query=query, llm=llm, count=variation_count):
        docs = retriever.invoke(search_query)

        for doc in docs:
            key = (
                doc.metadata.get('source_file'),
                doc.metadata.get('page_numer'),
                doc.metadata.get('chunk_id')
            )

            if key not in seen:
                seen.add(key)
                results.append(doc)

    return results

def create_retriever(vectorstore, top_k: int = DEFAULT_TOP_K):

    '''
    Creates a semantic retriever from te vector database
    '''

    return vectorstore.as_retriever(search_args = {'k':top_k})


def filter_by_section(docs: List[Document], section: str | None) -> List[Document]:

    '''
    Filters the chunks based upon the given section
    '''

    if section is None:
        return docs
    
    filterd_docs = [doc for doc in docs if doc.metadata.get('section') == section]

    return filterd_docs or docs


def hybrid_retrieve_with_query_variations(
        vectorstore,
        query: str,
        llm: ChatOpenAI,
        top_k: int,
        variation_count: int = 3,
        apply_section_filter: bool = False,
        diversify_sources: bool = False
) -> List[Document]:
    
    '''
    Combine Vector Search, Keyword Search, Query Variation and Selection Hints
    '''

    retriever = create_retriever(vectorstore, top_k)
    all_chunks = get_all_vectorstore_documents(vectorstore=vectorstore)
    section_filter = infer_section_filter(query=query)
    requested_sources = detect_requested_sources(query=query, docs=all_chunks)

    retrieved_docs=[]

    for search_query in create_query_variations(query=query,
                                                llm=llm,
                                                count=variation_count):
        vector_docs = retriever.invoke(search_query)
        keyword_docs = keyword_search(docs=all_chunks,
                                      query=search_query,
                                      top_k=top_k)
        retrieved_docs.extend(vector_docs + keyword_docs)


    unique_docs = deduplicate_docs(retrieved_docs)
    unique_docs = filter_by_requested_sources(docs=unique_docs, 
                                              requested_sources=requested_sources)
    final_docs = filter_by_section(docs = unique_docs, section=section_filter) if apply_section_filter else unique_docs
    
    if diversify_sources and not requested_sources:
        final_docs = diversify_by_sources(docs=final_docs)

    return final_docs[: top_k*5]