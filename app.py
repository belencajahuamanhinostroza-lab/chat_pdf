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
    page_title="SUPERZOO",
    page_icon="🐾",
    layout="wide"
)


# ==================================================
# ESTILOS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #666666;
        margin-bottom: 30px;
    }

    .animal-card {
        padding: 25px;
        border-radius: 20px;
        background-color: #f5f8f2;
        border: 1px solid #dfe7d8;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .answer-card {
        padding: 25px;
        border-radius: 20px;
        background-color: #f8faf7;
        border: 1px solid #dfe5da;
        margin-top: 15px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 15px;
        background-color: #ffffff;
        border: 1px solid #eeeeee;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
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

    st.header("🐾 SUPERZOO")

    st.write(
        "Tu experto virtual en animales."
    )

    st.divider()

    st.subheader("⚙️ Configuración")

    api_key = st.text_input(
        "Clave de OpenAI",
        type="password"
    )

    st.divider()

    st.markdown(
        """
        ### ¿Cómo funciona?

        📄 **1. Carga**  
        Sube información sobre un animal.

        🔎 **2. Analiza**  
        SUPERZOO procesa el documento.

        💬 **3. Pregunta**  
        Haz preguntas sobre el animal.

        🧠 **4. Responde**  
        SUPERZOO recupera información
        relevante y genera una respuesta.
        """
    )

    st.divider()

    if st.button(
        "🗑️ Nuevo animal",
        use_container_width=True
    ):

        st.session_state.knowledge_base = None
        st.session_state.chunks = []
        st.session_state.chat_history = []
        st.session_state.pdf_name = None
        st.session_state.pdf_pages = 0

        st.rerun()


# ==================================================
# HEADER
# ==================================================

col1, col2 = st.columns(
    [3, 1]
)

with col1:

    st.markdown(
        '<div class="main-title">🐾 SUPERZOO</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Tu experto virtual en animales'
        '</div>',
        unsafe_allow_html=True
    )


with col2:

    st.write("")


# ==================================================
# INTRODUCCIÓN
# ==================================================

if st.session_state.knowledge_base is None:

    st.info(
        "🔬 SUPERZOO analiza documentos sobre "
        "animales y responde tus preguntas "
        "utilizando la información recuperada."
    )


# ==================================================
# IMAGEN
# ==================================================

try:

    image = Image.open("Chat_pdf.png")

    st.image(
        image,
        width=250
    )

except Exception:

    pass


# ==================================================
# API KEY
# ==================================================

if not api_key:

    st.warning(
        "🔑 Ingresa tu clave de OpenAI "
        "en el panel lateral para comenzar."
    )


# ==================================================
# CARGAR PDF
# ==================================================

pdf = st.file_uploader(
    "📄 Carga información sobre un animal",
    type=["pdf"]
)


# ==================================================
# PROCESAR PDF
# ==================================================

if pdf is not None and api_key:

    os.environ["OPENAI_API_KEY"] = api_key

    if st.session_state.pdf_name != pdf.name:

        try:

            with st.spinner(
                "🔬 SUPERZOO está analizando el documento..."
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


                # Crear embeddings

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


                # Crear base vectorial

                knowledge_base = FAISS.from_texts(
                    texts,
                    embeddings,
                    metadatas=metadatas
                )


                # Guardar información

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
                "🐾 SUPERZOO terminó de analizar "
                "el documento."
            )


        except Exception as e:

            st.error(
                f"❌ No se pudo analizar el documento: {e}"
            )


# ==================================================
# DOCUMENTO CARGADO
# ==================================================

if st.session_state.knowledge_base is not None:

    st.markdown(
        f"""
        <div class="animal-card">

        <h2>🐾 Animal analizado</h2>

        <h3>
        {st.session_state.pdf_name}
        </h3>

        <p>
        📑 {st.session_state.pdf_pages} páginas
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🧩 {len(st.session_state.chunks)} fragmentos
        </p>

        <p>
        🧠 SUPERZOO está listo para responder
        preguntas sobre este documento.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ==================================================
    # ÁREAS DE CONOCIMIENTO
    # ==================================================

    st.subheader(
        "🔬 ¿Qué puedes preguntarle a SUPERZOO?"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class="feature-card">

            🧬

            <br>

            **Características**

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">

            🌎

            <br>

            **Hábitat**

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">

            🍖

            <br>

            **Alimentación**

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="feature-card">

            🐾

            <br>

            **Comportamiento**

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # ==================================================
    # HISTORIAL
    # ==================================================

    if st.session_state.chat_history:

        st.subheader(
            "💬 Conversación con SUPERZOO"
        )

        for item in st.session_state.chat_history:

            with st.chat_message("user"):

                st.write(
                    item["question"]
                )

            with st.chat_message("assistant"):

                st.write(
                    item["answer"]
                )

                with st.expander(
                    "📚 Información utilizada"
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
        "💬 Pregúntale a SUPERZOO"
    )

    user_question = st.text_area(
        "¿Qué quieres saber?",
        placeholder=(
            "Ejemplo: ¿Dónde vive este animal "
            "y de qué se alimenta?"
        ),
        height=100
    )


    # ==================================================
    # BOTÓN
    # ==================================================

    if st.button(
        "🔎 Preguntar a SUPERZOO",
        type="primary",
        use_container_width=True
    ):

        if not user_question.strip():

            st.warning(
                "Escribe una pregunta primero."
            )

        else:

            try:

                with st.spinner(
                    "🧠 SUPERZOO está buscando "
                    "la información..."
                ):

                    # Buscar información relevante

                    docs = (
                        st.session_state
                        .knowledge_base
                        .similarity_search(
                            user_question,
                            k=4
                        )
                    )


                    # Modelo

                    llm = OpenAI(
                        temperature=0,
                        model_name="gpt-4o-mini"
                    )


                    # Prompt del experto

                    expert_prompt = f"""
                    Eres SUPERZOO, un experto virtual
                    especializado en zoología y animales.

                    Tu función es responder preguntas
                    utilizando la información recuperada
                    del documento proporcionado.

                    REGLAS:

                    - Utiliza principalmente la información
                      encontrada en el documento.
                    - No inventes datos.
                    - Explica conceptos científicos de
                      manera clara.
                    - Si la información solicitada no aparece
                      en el documento, dilo claramente.
                    - No presentes información inventada
                      como si fuera un hecho.
                    - Responde como un experto en animales,
                      pero de manera comprensible para
                      cualquier usuario.

                    Pregunta del usuario:

                    {user_question}
                    """


                    chain = load_qa_chain(
                        llm,
                        chain_type="stuff"
                    )


                    response = chain.run(
                        input_documents=docs,
                        question=expert_prompt
                    )


                # Fuentes

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


                # Guardar conversación

                st.session_state.chat_history.append(
                    {
                        "question": user_question,
                        "answer": response,
                        "sources": sources
                    }
                )


                # Mostrar respuesta

                st.subheader(
                    "🧠 SUPERZOO responde"
                )

                st.markdown(
                    f"""
                    <div class="answer-card">

                    {response}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # Contexto

                with st.expander(
                    "📚 Ver información utilizada"
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
                    f"❌ SUPERZOO encontró un error: {e}"
                )


# ==================================================
# ESTADO INICIAL
# ==================================================

elif pdf is None and api_key:

    st.info(
        "📄 Carga un PDF para que SUPERZOO "
        "pueda comenzar su análisis."
    )
