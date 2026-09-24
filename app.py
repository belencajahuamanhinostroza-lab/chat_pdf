import os
import io
import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from openai import OpenAI as OpenAIClient


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="RAG - Pregúntale a tu PDF",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #777;
    font-size: 17px;
    margin-bottom: 25px;
}

.document-card {
    background-color: #f5f7fa;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e5e9;
    margin-bottom: 20px;
}

.question-box {
    background-color: #f8f9fb;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e1e4e8;
}

.answer-box {
    background-color: #eef6ff;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #3182ce;
}

.context-box {
    background-color: #fafafa;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #ddd;
}

.small-text {
    color: #777;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TÍTULO
# ============================================================

st.markdown(
    '<div class="main-title">📚 RAG — Pregúntale a tu PDF</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Carga un documento, haz preguntas y descubre qué información utilizó la IA para responder.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

api_key = None

# Primero intenta obtener la clave desde Streamlit Secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass


# Si no existe en Secrets, permite ingresarla manualmente
if not api_key:

    with st.sidebar:

        st.header("⚙️ Configuración")

        api_key = st.text_input(
            "Clave de OpenAI",
            type="password",
            help="Solo necesaria si no configuraste OPENAI_API_KEY en Streamlit Secrets."
        )

        st.caption(
            "Recomendación: para una aplicación publicada, "
            "guarda la clave en Streamlit Secrets."
        )


if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Documento")

    st.write(
        "Sube un PDF y podrás hacer preguntas sobre su contenido."
    )

    if st.button("🗑️ Nuevo documento", use_container_width=True):

        keys_to_delete = [
            "knowledge_base",
            "pdf_name",
            "pdf_pages",
            "chunks",
            "chat_history",
            "last_context",
            "audio"
        ]

        for key in keys_to_delete:

            if key in st.session_state:
                del st.session_state[key]

        st.rerun()


# ============================================================
# ESTADOS DE SESIÓN
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "last_context" not in st.session_state:
    st.session_state.last_context = []

if "audio" not in st.session_state:
    st.session_state.audio = None


# ============================================================
# SUBIR PDF
# ============================================================

pdf = st.file_uploader(
    "📎 Carga tu archivo PDF",
    type=["pdf"]
)


# ============================================================
# PROCESAMIENTO DEL PDF
# ============================================================

if pdf is not None and api_key:

    # Evita procesar nuevamente el mismo PDF en cada interacción
    if st.session_state.pdf_name != pdf.name:

        try:

            with st.spinner("📖 Analizando el documento..."):

                pdf_reader = PdfReader(pdf)

                pages_text = []

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):

                    page_text = page.extract_text() or ""

                    pages_text.append({
                        "page": page_number,
                        "text": page_text
                    })


                # ------------------------------------------------
                # Crear fragmentos conservando la página
                # ------------------------------------------------

                documents = []

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=700,
                    chunk_overlap=100,
                    length_function=len
                )


                for page_data in pages_text:

                    chunks = text_splitter.split_text(
                        page_data["text"]
                    )

                    for chunk in chunks:

                        documents.append(
                            {
                                "text": chunk,
                                "page": page_data["page"]
                            }
                        )


                texts = [
                    item["text"]
                    for item in documents
                ]


                # ------------------------------------------------
                # Embeddings
                # ------------------------------------------------

                embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key
                )


                knowledge_base = FAISS.from_texts(
                    texts,
                    embeddings,
                    metadatas=[
                        {
                            "page": item["page"]
                        }
                        for item in documents
                    ]
                )


                # Guardar en sesión
                st.session_state.knowledge_base = knowledge_base

                st.session_state.pdf_name = pdf.name

                st.session_state.pdf_pages = len(
                    pdf_reader.pages
                )

                st.session_state.chunks = documents

                st.session_state.chat_history = []

                st.session_state.last_context = []

                st.session_state.audio = None


            st.success("✅ Documento procesado correctamente.")


        except Exception as e:

            st.error(
                f"❌ Error al procesar el PDF: {str(e)}"
            )


# ============================================================
# INFORMACIÓN DEL DOCUMENTO
# ============================================================

if st.session_state.knowledge_base is not None:

    st.markdown("### 📄 Documento cargado")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documento",
            st.session_state.pdf_name
        )

    with col2:

        st.metric(
            "Páginas",
            st.session_state.pdf_pages
        )

    with col3:

        st.metric(
            "Fragmentos",
            len(st.session_state.chunks)
        )


    # ========================================================
    # CHAT ANTERIOR
    # ========================================================

    if st.session_state.chat_history:

        st.markdown("### 💬 Conversación")

        for item in st.session_state.chat_history:

            with st.chat_message("user"):

                st.write(item["question"])


            with st.chat_message("assistant"):

                st.markdown(item["answer"])

                if item.get("audio"):

                    st.audio(
                        item["audio"],
                        format="audio/mp3"
                    )


    # ========================================================
    # PREGUNTA
    # ========================================================

    st.markdown("---")

    st.markdown("### 🔎 Pregúntale al documento")

    user_question = st.text_input(
        "Pregunta",
        placeholder="Ejemplo: ¿Dónde vive el león?",
        label_visibility="collapsed"
    )


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
                    "🔎 Buscando información relevante..."
                ):

                    # ------------------------------------------------
                    # BÚSQUEDA SEMÁNTICA
                    # ------------------------------------------------

                    docs = (
                        st.session_state
                        .knowledge_base
                        .similarity_search(
                            user_question,
                            k=4
                        )
                    )


                    # ------------------------------------------------
                    # GUARDAR CONTEXTO
                    # ------------------------------------------------

                    context_data = []

                    for doc in docs:

                        page = doc.metadata.get(
                            "page",
                            "?"
                        )

                        context_data.append(
                            {
                                "page": page,
                                "text": doc.page_content
                            }
                        )


                    st.session_state.last_context = (
                        context_data
                    )


                    # ------------------------------------------------
                    # MODELO
                    # ------------------------------------------------

                    llm = OpenAI(
                        temperature=0,
                        model_name="gpt-4o-mini",
                        openai_api_key=api_key
                    )


                    # ------------------------------------------------
                    # PROMPT RAG
                    # ------------------------------------------------

                    chain = load_qa_chain(
                        llm,
                        chain_type="stuff"
                    )


                    response = chain.run(
                        input_documents=docs,
                        question=(
                            "Responde únicamente utilizando "
                            "la información contenida en el "
                            "documento proporcionado. "
                            "Si la respuesta no aparece "
                            "en el documento, indica claramente "
                            "que esa información no se encuentra "
                            "en el PDF.\n\n"
                            f"Pregunta: {user_question}"
                        )
                    )


                    # ------------------------------------------------
                    # AUDIO
                    # ------------------------------------------------

                    audio_data = None

                    try:

                        client = OpenAIClient(
                            api_key=api_key
                        )


                        speech = client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="coral",
                            input=response
                        )


                        audio_data = speech.content


                    except Exception as audio_error:

                        st.warning(
                            "La respuesta se generó correctamente, "
                            "pero no se pudo crear el audio."
                        )


                    # ------------------------------------------------
                    # GUARDAR CHAT
                    # ------------------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "question": user_question,
                            "answer": response,
                            "audio": audio_data
                        }
                    )


                    st.rerun()


            except Exception as e:

                st.error(
                    f"❌ Error al generar la respuesta: {str(e)}"
                )


    # ========================================================
    # ÚLTIMO CONTEXTO RECUPERADO
    # ========================================================

    if st.session_state.last_context:

        st.markdown("---")

        with st.expander(
            "🔎 Ver contexto utilizado por el RAG"
        ):

            st.caption(
                "Estos son los fragmentos del PDF que "
                "el sistema recuperó para construir la respuesta."
            )


            for index, item in enumerate(
                st.session_state.last_context,
                start=1
            ):

                st.markdown(
                    f"**Fragmento {index} · Página {item['page']}**"
                )

                st.markdown(
                    f"> {item['text']}"
                )

                st.divider()


# ============================================================
# ESTADO INICIAL
# ============================================================

elif not api_key:

    st.info(
        "🔐 Ingresa tu API Key o configura "
        "`OPENAI_API_KEY` en Streamlit Secrets."
    )

else:

    st.info(
        "📎 Carga un PDF para comenzar."
    )
