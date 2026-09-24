import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import platform


# ==================================================
# CONFIGURACIÓN
# ==================================================

st.set_page_config(
    page_title="RAG - Pregúntale a tu PDF",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# ESTILOS
# ==================================================

st.markdown(
    """
    <style>
    
    .document-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f5f5;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }

    .answer-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f7f9fc;
        border: 1px solid #d9dee8;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# TÍTULO
# ==================================================

st.title("📚 RAG — Pregúntale a tu PDF")

st.caption(
    "Generación Aumentada por Recuperación"
)

st.write(
    "Versión de Python:",
    platform.python_version()
)


# ==================================================
# SESSION STATE
# ==================================================

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.subheader("⚙️ Configuración")

    st.write(
        "Carga un PDF y realiza preguntas "
        "sobre su contenido."
    )

    ke = st.text_input(
        "Clave de OpenAI",
        type="password"
    )

    st.divider()

    if st.button(
        "🗑️ Nuevo documento",
        use_container_width=True
    ):

        st.session_state.knowledge_base = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.pdf_name = None
        st.session_state.pdf_pages = 0

        st.rerun()


# ==================================================
# API KEY
# ==================================================

if ke:

    os.environ["OPENAI_API_KEY"] = ke

else:

    st.warning(
        "🔑 Ingresa tu clave de API de OpenAI "
        "para continuar."
    )


# ==================================================
# IMAGEN
# ==================================================

try:

    image = Image.open("Chat_pdf.png")

    st.image(
        image,
        width=300
    )

except Exception:

    pass


# ==================================================
# CARGAR PDF
# ==================================================

pdf = st.file_uploader(
    "📄 Carga tu archivo PDF",
    type=["pdf"]
)


# ==================================================
# PROCESAR PDF
# ==================================================

if pdf is not None and ke:

    if st.session_state.pdf_name != pdf.name:

        try:

            with st.spinner(
                "📖 Analizando el documento..."
            ):

                pdf_reader = PdfReader(pdf)

                total_pages = len(
                    pdf_reader.pages
                )

                all_chunks = []

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):

                    page_text = page.extract_text()

                    if not page_text:
                        continue

                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len
                    )

                    page_chunks = (
                        text_splitter.split_text(
                            page_text
                        )
                    )

                    for chunk in page_chunks:

                        all_chunks.append(
                            {
                                "text": chunk,
                                "page": page_number
                            }
                        )


                # ----------------------------------
                # CREAR EMBEDDINGS
                # ----------------------------------

                embeddings = OpenAIEmbeddings()

                texts = [
                    chunk["text"]
                    for chunk in all_chunks
                ]

                metadatas = [
                    {
                        "page": chunk["page"]
                    }
                    for chunk in all_chunks
                ]


                # ----------------------------------
                # CREAR BASE VECTORIAL
                # ----------------------------------

                knowledge_base = FAISS.from_texts(
                    texts,
                    embeddings,
                    metadatas=metadatas
                )


                # ----------------------------------
                # GUARDAR INFORMACIÓN
                # ----------------------------------

                st.session_state.knowledge_base = (
                    knowledge_base
                )

                st.session_state.chunks = (
                    all_chunks
                )

                st.session_state.pdf_name = (
                    pdf.name
                )

                st.session_state.pdf_pages = (
                    total_pages
                )

                st.session_state.chat_history = []


            st.success(
                "✅ Documento procesado correctamente."
            )


        except Exception as e:

            st.error(
                f"❌ Error al procesar el PDF: {e}"
            )


# ==================================================
# INFORMACIÓN DEL DOCUMENTO
# ==================================================

if st.session_state.knowledge_base is not None:

    st.markdown(
        f"""
        <div class="document-card">
            <h3>📄 {st.session_state.pdf_name}</h3>

            <p>
                📑 Páginas:
                <b>{st.session_state.pdf_pages}</b>
                &nbsp;&nbsp;&nbsp;
                🧩 Fragmentos:
                <b>{len(st.session_state.chunks)}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ==================================================
    # HISTORIAL
    # ==================================================

    if st.session_state.chat_history:

        st.subheader("💬 Conversación")

        for item in st.session_state.chat_history:

            with st.chat_message("user"):

                st.write(
                    item["question"]
                )

            with st.chat_message("assistant"):

                st.write(
                    item["answer"]
                )

                if item["sources"]:

                    with st.expander(
                        "📚 Ver contexto utilizado"
                    ):

                        for source in item["sources"]:

                            st.markdown(
                                f"""
                                **Página {source["page"]}**

                                {source["text"]}
                                """
                            )


    # ==================================================
    # PREGUNTA
    # ==================================================

    st.subheader(
        "🔎 ¿Qué quieres saber sobre el documento?"
    )

    user_question = st.text_area(
        "Escribe tu pregunta:",
        placeholder=(
            "Ejemplo: ¿Cuál es el tema principal "
            "del documento?"
        ),
        height=100
    )


    # ==================================================
    # BOTÓN PREGUNTAR
    # ==================================================

    if st.button(
        "➤ Preguntar",
        type="primary",
        use_container_width=True
    ):

        if not user_question.strip():

            st.warning(
                "Escribe una pregunta antes de continuar."
            )

        else:

            try:

                with st.spinner(
                    "🤔 Buscando la información..."
                ):

                    # ----------------------------------
                    # BUSCAR CONTEXTO
                    # ----------------------------------

                    docs = (
                        st.session_state
                        .knowledge_base
                        .similarity_search(
                            user_question,
                            k=4
                        )
                    )


                    # ----------------------------------
                    # MODELO
                    # ----------------------------------

                    llm = OpenAI(
                        temperature=0,
                        model_name="gpt-4o-mini"
                    )


                    # ----------------------------------
                    # CADENA RAG
                    # ----------------------------------

                    chain = load_qa_chain(
                        llm,
                        chain_type="stuff"
                    )


                    # ----------------------------------
                    # RESPUESTA
                    # ----------------------------------

                    response = chain.run(
                        input_documents=docs,
                        question=(
                            "Responde utilizando "
                            "únicamente la información "
                            "del documento.\n\n"
                            "Si la respuesta no se encuentra "
                            "en el documento, indica que "
                            "no se encontró información "
                            "suficiente.\n\n"
                            f"Pregunta: {user_question}"
                        )
                    )


                # ----------------------------------
                # GUARDAR FUENTES
                # ----------------------------------

                sources = []

                for doc in docs:

                    sources.append(
                        {
                            "text": doc.page_content,
                            "page": doc.metadata.get(
                                "page",
                                "desconocida"
                            )
                        }
                    )


                # ----------------------------------
                # GUARDAR CHAT
                # ----------------------------------

                st.session_state.chat_history.append(
                    {
                        "question": user_question,
                        "answer": response,
                        "sources": sources
                    }
                )


                # ----------------------------------
                # MOSTRAR RESPUESTA
                # ----------------------------------

                st.subheader(
                    "💡 Respuesta"
                )

                st.markdown(
                    f"""
                    <div class="answer-card">
                        {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ----------------------------------
                # CONTEXTO
                # ----------------------------------

                with st.expander(
                    "📚 Ver contexto recuperado"
                ):

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"### Fragmento {i}"
                        )

                        st.markdown(
                            f"**Página {source['page']}**"
                        )

                        st.write(
                            source["text"]
                        )

                        st.divider()


            except Exception as e:

                st.error(
                    f"❌ Ocurrió un error: {e}"
                )


# ==================================================
# MENSAJE INICIAL
# ==================================================

elif pdf is None:

    st.info(
        "📄 Carga un PDF para comenzar."
    )

elif not ke:

    st.warning(
        "🔑 Ingresa tu clave de OpenAI para continuar."
    )
