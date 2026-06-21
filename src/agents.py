# agents.py

'''
Agentic RAG that adds a decision step:
The system checks if the retrieved context is enough before generating
'''

from typing import Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.config import CHAT_MODEL
from src.retriever import format_context, hybrid_retrieve_with_query_variations


class AgentState(TypedDict, total=False):
    'Passes memeory between graph nodes'

    query: str
    retrieved_docs: list
    refined_query: str
    answer: str
    sources: List[Dict]
    needs_retry: bool
    summary: str


def create_llm(temperature: float = 0) -> ChatOpenAI:

    '''
    Creates a chat model used for grading, rewriting query and amswering
    '''
    return ChatOpenAI(model=CHAT_MODEL, temperature=temperature)


def grade_retrieval_quality(query: str, context: str, llm: ChatOpenAI) -> bool:

    '''
    Returns True if the context is sufficient to answer the query else False
    '''

    prompt= f"""
You are checking whether the retrieved research context is sufficient to answer the query.

Query:
{query}

Retrieved Context:
{context}

Answer only YES or NO.
"""
    response = llm.invoke(prompt).content.strip().lower()
    
    return response.startswith('yes')


def refine_query(query: str, llm: ChatOpenAI) -> str:

    '''
    Rewrites a query after refinement for better retrieval
    '''

    prompt= f"""
Rewrite this query as a stronger search query to get a better retrieval for clinical research papers.
Keep the original meaning as is.

Query:
{query}
"""
    
    return llm.invoke(prompt).content.strip()


def generate_answer(query: str, docs: list, llm: ChatOpenAI) -> str:

    '''
    Generates a grounded answer using only retrieved context
    '''

    context = format_context(docs=docs)

    prompt= f"""
You are an AI Clinical Research Assistant.
Answer the questions using ONLY the provided research context.

Important Rules:
- Cite the sources like [Source 1], [Source 2]
- Do not invent fact
- If the answer is not present, say: `Not Found in the provided research papers.`
- Use a precise academic style

Context:
{context}

Query:
{query}
"""
    
    return llm.invoke(prompt).content


def generate_paper_summary(query: str, docs: list, llm: ChatOpenAI) -> str:

    '''
    Generates a structured summary using the provided context only from the research papers.
    '''

    context = format_context(docs=docs)

    prompt = f"""
You are a Clinical Research Assistant.

Create a structured summary using ONLY the provided context.

Use this fomat:
1. - Research focus
2. - Papers/Studies covered
3. - Methods and Population Details
4. - Main Findings
5. - Limitations
6. - Practical clinical interpretations

Important Rules:
- Cite the sources like [Source 1], [Source 2]
- Do not invent fact
- If the answer is not present, say: `Not Found in the provided research papers.`
- Use a precise academic style

Context:
{context}

Query:
{query}
"""
    
    return llm.invoke(prompt).content

def extract_sources(docs: list) -> List[Dict]:

    '''
    Creates a clean source metadata for the final UI reports
    '''

    sources = []

    for doc in docs:
        source ={
            'study': doc.metadata.get('author_year', 'Unknown'),
            'file': doc.metadata.get('source_file', 'Unknown'),
            'page': doc.metadata.get('page_number', 'NA'),
            'chunk_id': doc.metadata.get('chunk_id', 'NA')
        }

        if source not in sources:
            sources.append(source)

    return sources

def build_agentic_rag_graph(vectorstore, top_k: int = 5):
    
    '''
    Builds a langgrph workflow with retrieve, grade, retry and answer nodes
    '''

    llm = create_llm()

    def retrieve_node(state: AgentState) -> AgentState:
        docs = hybrid_retrieve_with_query_variations(vectorstore=vectorstore,
                                                     query=state['query'],
                                                     llm=llm,
                                                     top_k=top_k,
                                                     )
        
        return {'retrieved_docs': docs}
    
    def grade_node(state: AgentState) -> AgentState:
        context = format_context(docs=state['retrieved_docs'])
        is_enouh = grade_retrieval_quality(query=state['query'], 
                                           context=context,
                                           llm=llm)
        
        return {
            'needs_retry': not is_enouh,
            'refined_query': state['query'] if is_enouh else refine_query(query=state['query'], llm=llm)
        }
    
    def retrieve_again_node(state: AgentState) -> AgentState:
        docs = hybrid_retrieve_with_query_variations(
            vectorstore=vectorstore,
            query=state['refined_query'],
            llm=llm,
            top_k=top_k
        )
        
        return {'retrieved_docs': docs}
    
    def generate_node(state: AgentState) -> AgentState:
        docs = state['retrieved_docs']

        return {
            'answer': generate_answer(query=state['query'], 
                                      docs=docs, 
                                      llm=llm),
            'sources': extract_sources(docs=docs)
        }
    

    def decide_next_step(state: AgentState) -> str:
        return 'retrieve_again' if state.get('needs_retry') else 'generate'
    

    graph = StateGraph(AgentState)
    graph.add_node('retrieve', retrieve_node)
    graph.add_node('grade', grade_node)
    graph.add_node('retrieve_again', retrieve_again_node)
    graph.add_node('generate', generate_node)

    graph.set_entry_point('retrieve')
    graph.add_edge('retrieve', 'grade')
    graph.add_conditional_edges(
        'grade',
        decide_next_step,
        {
            'retrieve_again': 'retrieve_again',
            'generate': 'generate'
        }
    )
    graph.add_edge('retrieve_again', 'generate')
    graph.add_edge('generate', END)

    return graph.compile()


def build_paper_summary_graph(vectorstore, top_k: int =8):
    
    '''
    Builds a Langgraph workflow for summarising indexed papers.
    '''
    llm = create_llm()

    def retrieve_for_summary_node(state: AgentState) -> AgentState:
        docs = hybrid_retrieve_with_query_variations(
            vectorstore=vectorstore,
            query=state['query'],
            llm=llm,
            top_k=top_k,
            apply_section_filter=False,
            diversify_sources=True
        )

        return {'retrieved_docs': docs}
    
    def summarize_node(state: AgentState) -> AgentState:
        docs = state['retrieved_docs']

        return {
            'summary': generate_paper_summary(query=state['query'],
                                              docs=docs,
                                              llm=llm),
            'sources': extract_sources(docs=docs)
        }
    

    graph = StateGraph(AgentState)
    graph.add_node('retrieve_for_summary', retrieve_for_summary_node)
    graph.add_node('summarize', summarize_node)

    graph.set_entry_point('retrieve_for_summary')
    graph.add_edge('retrieve_for_summary', 'summarize')
    graph.add_edge('summarize', END)

    return graph.compile()


