# app.py

"""Streamlit app for the Clinical Research Assistant"""


from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    PDF_DIR,
    UPLOAD_DIR
)


from src.pdf_loader import list_pdf_files
from src.services import (
    ClinicalResearchAssistant,
    PaperIndexingService,
    SectionChunker,
    PaperIndexingService,
    load_existing_vectorstore
)

load_dotenv()

st.set_page_config(page_title = "Clinical Research Assistant", layout = 'wide')

def save_uploaded_files(uploaded_files, upload_dir:Path) -> list[Path]:
    """Save Streamlit uploads to the ingestion pipeline"""

    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths=[]

    for uploaded_file in uploaded_files:
        file_path = upload_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def index_paper(pdf_paths: list[Path],
                chunk_size:int,
                chunk_overlap:int):
    """Run the complete indexing pipeline: load, chunk, embed and store"""

    service = PaperIndexingService(
        chunker=SectionChunker(
        chunk_overlap = chunk_overlap,
        chunk_size = chunk_size
        )
    )

    return service.index(pdf_paths=pdf_paths)

def get_vectorstore():
    """Loads the vector databases if it has already been created"""

    return load_existing_vectorstore()


st.title("Clinical Research Assistant")

st.sidebar.header('Index Papers')
chunk_size = st.sidebar.slider('Chunk Size', 400, 1600, DEFAULT_CHUNK_SIZE, 100)
chunk_overlap = st.sidebar.slider('Chunk Overlap', 0, 300, DEFAULT_CHUNK_OVERLAP, 20)
top_k = st.sidebar.slider('Top-k per query', 2, 10, DEFAULT_TOP_K, 1)

uploaded_files = st.sidebar.file_uploader(
    'Upload clinical research pdfs',
    type = ['pdf'],
    accept_multiple_files = True
)

use_existing_pdfs = st.sidebar.checkbox(
    "Also index PDFs from the existing pdf folder",
    value = True
    )

if st.sidebar.button('Build / Rebuild Research Index'):

    pdf_paths =[]

    if use_existing_pdfs:
        pdf_paths.extend(list_pdf_files(PDF_DIR))

    if uploaded_files:
        pdf_paths.extend(save_uploaded_files(uploaded_files, UPLOAD_DIR))

    if not pdf_paths:
        st.sidebar.error("Please upload PDFs or keep the exising PDF folder selected")
    else:
        with st.spinner('Reading PDFs, creating chunks, and building Vector DB...'):
            result = index_paper(pdf_paths = pdf_paths,
                                 chunk_size = chunk_size,
                                 chunk_overlap = chunk_overlap)
            
        st.session_state['vectorstore'] = result.vectorstore
        st.sidebar.success(
            f"Indexed {result.pages} pages into {result.chunks} chunks"
        )


vectorstore = st.session_state.get('vectorstore') or get_vectorstore()
assistant = ClinicalResearchAssistant(vectorstore = vectorstore,top_k = top_k)  if vectorstore else None

main, right = st.columns([2, 1])

with main:
    ask_tab, summary_tab = st.tabs(['Ask Questions', 'Summarize Papers'])

    with ask_tab:
        st.subheader('Ask a Research Question')
        question = st.text_area(
            "Question",
            placeholder = "Example: What ae the cardiovascular outcomes reported in xxx study?",
            height = 100
        )

        if st.button("Ask Assistant", type ='primary'):
            if assistant is None:
                st.error('Build the research index first')
            elif not question.strip():
                st.error("Enter a research question")
            else:
                with st.spinner('Retrieving evidence, checking quality and generating answer...'):
                    response = assistant.ask(question=question)

                    st.markdown("### Answer")
                    st.write(response['answer'])

                    st.markdown('### Sources')
                    for source in response.get("sources", []):
                        st.write(
                            f"- {source['study']} | {source['file']} | "
                            f"Page {source['page']} | Chunk {source['chunk_id']}"
                        )

    
    with summary_tab:
        st.subheader("Summarise Indexed Papers")
        summary_focus = st.text_input(
            "Summary Focus",
            value = "Summarize the clinical objective, population, methods, findings and limitations",
        )

        if st.button("Generate Summary"):
            if assistant is None:
                st.error("Build the research index first")
            else:
                with st.spinner('Retrieving paper evidence and generating summary...'):
                    response = assistant.summarise(focus=summary_focus)

                st.markdown('### Summary')
                st.write(response['summary'])

                st.markdown('### Sources')
                for source in response.get('sources', []):
                    st.write(
                        f"- {source['study']} | {source['file']} | "
                        f"Page {source['page']} | Chunk {source['chunk_id']}"
                    )


with right:
    st.subheader("System Workflow")
    st.write('1. Upload PDFs')
    st.write('2. Extract page text')
    st.write('3. Split pages into chunks')
    st.write('4. Create embeddings')
    st.write('5. Store chunks in ChromaDB')
    st.write('6. Create query variations')
    st.write('7. Retrieve evidence')
    st.write('8. Grade evidence quality')
    st.write('9. Retry if weak context')
    st.write('10. Answer with citations')
