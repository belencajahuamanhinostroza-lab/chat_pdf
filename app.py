import os
import base64
import streamlit as st

from PIL import Image
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="SUPERZOO",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ESTILOS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 45%, #a5d6a7 100%);
}

.main {
    background-color: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* HEADER */

.super-header {
    background: linear-gradient(135deg, #14532d, #166534, #15803d);
    padding: 30px;
    border-radius: 25px;
    color: white;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.super-header h1 {
    font-size: 52px;
    margin-bottom: 5px;
    font-weight: 800;
}

.super-header p {
    font-size: 18px;
    margin-top: 5px;
}

/* TARJETAS */

.info-card {
    background: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(22,101,52,0.15);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.info-card h3 {
    color: #14532d;
    margin-bottom: 8px;
}

/* BLOQUES */

.block-card {
    background: #f7fff8;
    border-left: 6px solid #22c55e;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.block-title {
    font-weight: bold;
    color: #166534;
    font-size: 17px;
}

/* CHAT */

.question-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
}

.answer-card {
    background: #f0fdf4;
    border-left: 6px solid #16a34a;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
}

/* BOTONES */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    border: none;
    min-height: 45px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_info" not in st.session_state:
    st.session_state.chunk_info = []

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_image_result" not in st.session_state:
    st.session_state.last_image_result = ""


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="super-header">

<h1>🐾 SUPERZOO 🦁</h1>

<p>
Tu experto virtual en animales, hábitats y comportamiento
</p>

<div style="font-size:32px; margin-top:12px;">
🌿 🦜 🐘 🐍 🐼 🐯 🌱 🦋
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌿 SUPERZOO")

    st.write(
        "Carga información sobre animales y permite que "
        "SUPERZOO la analice y responda tus preguntas."
    )

    st.markdown("---")

    st.markdown("### 🔑 OpenAI API Key")

    api_key = st.text_input(
        "Ingresa tu clave",
        type="password"
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")

    st.markdown("### 🐾 Áreas de conocimiento")

    st.write("🧬 Características")
    st.write("🌎 Hábitat")
    st.write("🍖 Alimentación")
    st.write("🐾 Comportamiento")
    st.write("🧡 Conservación")


# =========================================================
# VERIFICACIÓN API
# =========================================================

if not api_key:

    st.info(
        "🔑 Ingresa tu API Key en el panel izquierdo para comenzar."
    )

    st.stop()


# =========================================================
# CARGAR PDF
# =========================================================

st.markdown("## 📚 Base de conocimiento")

uploaded_pdf = st.file_uploader(
    "Carga un documento PDF sobre animales",
    type=["pdf"]
)


if uploaded_pdf is not None:

    # Evitar reprocesar el mismo archivo constantemente
    if st.session_state.file_name != uploaded_pdf.name:

        try:

            with st.spinner("🧠 SUPERZOO está leyendo el documento..."):

                pdf_reader = PdfReader(uploaded_pdf)

                full_text = ""

                page_texts = []

                for page_number, page in enumerate(pdf_reader.pages, start=1):

                    page_text = page.extract_text() or ""

                    page_texts.append(page_text)

                    full_text += (
                        f"\n\n[PÁGINA {page_number}]\n"
                        f"{page_text}"
                    )

                # -------------------------------------------------
                # DIVISIÓN EN BLOQUES
                # -------------------------------------------------

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=700,
                    chunk_overlap=100,
                    length_function=len
                )

                chunks = text_splitter.split_text(full_text)

                # -------------------------------------------------
                # EMBEDDINGS
                # -------------------------------------------------

                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small"
                )

                knowledge_base = FAISS.from_texts(
                    chunks,
                    embeddings
                )

                # -------------------------------------------------
                # GUARDAR INFORMACIÓN
                # -------------------------------------------------

                st.session_state.knowledge_base = knowledge_base
                st.session_state.chunks = chunks
                st.session_state.file_name = uploaded_pdf.name
                st.session_state.pages = len(pdf_reader.pages)

                # Crear resumen visual de cada bloque
                chunk_info = []

                for i, chunk in enumerate(chunks):

                    clean_chunk = chunk.replace(
                        "\n",
                        " "
                    ).strip()

                    if len(clean_chunk) > 220:
                        preview = clean_chunk[:220] + "..."
                    else:
                        preview = clean_chunk

                    chunk_info.append({
                        "numero": i + 1,
                        "texto": preview
                    })

                st.session_state.chunk_info = chunk_info

                st.success(
                    "✅ Documento procesado correctamente."
                )

        except Exception as e:

            st.error(
                f"❌ No se pudo procesar el PDF: {str(e)}"
            )

    # =========================================================
    # INFORMACIÓN DEL DOCUMENTO
    # =========================================================

    st.markdown("### 📄 Información del documento")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 Páginas",
            st.session_state.pages
        )

    with col2:
        st.metric(
            "🧩 Bloques",
            len(st.session_state.chunks)
        )

    with col3:
        st.metric(
            "📚 Fuente",
            "PDF"
        )


    # =========================================================
    # INFORMACIÓN DE LOS BLOQUES
    # =========================================================

    with st.expander(
        "🔎 Ver qué información contiene cada bloque",
        expanded=False
    ):

        st.write(
            "Estos son los fragmentos en los que SUPERZOO "
            "dividió el documento para poder recuperar "
            "información relevante."
        )

        for info in st.session_state.chunk_info:

            st.markdown(
                f"""
                <div class="block-card">

                <div class="block-title">
                🧩 Bloque {info["numero"]}
                </div>

                <div style="margin-top:8px;">
                {info["texto"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PREGUNTAS AL DOCUMENTO
# =========================================================

if st.session_state.knowledge_base is not None:

    st.markdown("---")

    st.markdown("""
    <div class="question-card">

    <h2>🧠 Pregúntale a SUPERZOO</h2>

    <p>
    Haz preguntas sobre los animales presentes en el documento.
    </p>

    </div>
    """, unsafe_allow_html=True)

    with st.form("question_form"):

        user_question = st.text_area(
            "💬 Tu pregunta",
            placeholder=(
                "Ejemplo: ¿Dónde vive este animal y "
                "de qué se alimenta?"
            ),
            height=100
        )

        send_question = st.form_submit_button(
            "📨 Enviar pregunta",
            use_container_width=True
        )

    if send_question:

        if not user_question.strip():

            st.warning(
                "✏️ Escribe una pregunta antes de enviarla."
            )

        else:

            try:

                with st.spinner(
                    "🧠 SUPERZOO está buscando información..."
                ):

                    # ---------------------------------------------
                    # RETRIEVAL
                    # ---------------------------------------------

                    docs = st.session_state.knowledge_base.similarity_search(
                        user_question,
                        k=4
                    )

                    retrieved_text = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    # ---------------------------------------------
                    # MODELO
                    # ---------------------------------------------

                    llm = ChatOpenAI(
                        model="gpt-5.6-luna",
                        temperature=0
                    )

                    prompt = f"""
Eres SUPERZOO, un experto virtual especializado
en zoología, animales, biodiversidad y conservación.

Tu objetivo es responder preguntas utilizando
PRINCIPALMENTE la información recuperada del documento.

REGLAS:

1. No inventes información.
2. Si la respuesta aparece en el documento,
   utiliza esa información.
3. Explica los conceptos científicos de forma clara.
4. Puedes organizar la respuesta con listas cuando ayude.
5. Si la información no aparece en el documento,
   dilo claramente.
6. No afirmes que un dato está en el documento
   si realmente no aparece.
7. Responde siempre en español.
8. Mantén un tono de experto pero fácil de entender.

INFORMACIÓN RECUPERADA DEL DOCUMENTO:

{retrieved_text}

PREGUNTA DEL USUARIO:

{user_question}

RESPUESTA DE SUPERZOO:
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })

                    # ---------------------------------------------
                    # RESPUESTA
                    # ---------------------------------------------

                    st.markdown("""
                    <div class="answer-card">

                    <h3>🐾 SUPERZOO responde</h3>

                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(answer)

                    # ---------------------------------------------
                    # INFORMACIÓN UTILIZADA
                    # ---------------------------------------------

                    with st.expander(
                        "📚 Ver información utilizada para responder"
                    ):

                        st.write(
                            "SUPERZOO recuperó estos fragmentos "
                            "del documento para construir la respuesta:"
                        )

                        for i, doc in enumerate(docs):

                            st.markdown(
                                f"""
                                <div class="block-card">

                                <div class="block-title">
                                🔎 Información recuperada {i + 1}
                                </div>

                                <div style="margin-top:8px;">
                                {doc.page_content}
                                </div>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            except Exception as e:

                st.error(
                    f"❌ Ocurrió un error al generar la respuesta: {str(e)}"
                )


# =========================================================
# HISTORIAL
# =========================================================

if st.session_state.chat_history:

    st.markdown("---")

    st.markdown("## 💬 Historial de preguntas")

    for item in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"""
            <div class="info-card">

            <strong>🧑 Pregunta:</strong>

            <p>
            {item["question"]}
            </p>

            <strong>🐾 SUPERZOO:</strong>

            <p>
            {item["answer"]}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ANÁLISIS DE IMÁGENES
# =========================================================

st.markdown("---")

st.markdown("## 📷 Analiza un animal")

st.write(
    "Sube una fotografía y SUPERZOO analizará "
    "lo que aparece en ella."
)

uploaded_image = st.file_uploader(
    "Sube una imagen de un animal",
    type=["jpg", "jpeg", "png"],
    key="animal_image"
)


if uploaded_image:

    col_img, col_text = st.columns([1, 1])

    with col_img:

        st.image(
            uploaded_image,
            caption=uploaded_image.name,
            use_column_width=True
        )

    with col_text:

        with st.form("image_form"):

            image_question = st.text_area(
                "🔎 ¿Qué quieres saber de la imagen?",
                placeholder=(
                    "Ejemplo: Describe sus características "
                    "físicas y dime qué tipo de animal parece ser."
                )
            )

            analyze_image = st.form_submit_button(
                "🐾 Analizar imagen",
                use_container_width=True
            )

        if analyze_image:

            if not image_question.strip():

                image_question = (
                    "Identifica y describe el animal de la imagen. "
                    "Explica sus características físicas, "
                    "posible hábitat y comportamiento."
                )

            try:

                with st.spinner(
                    "🔬 SUPERZOO está analizando la imagen..."
                ):

                    image_bytes = uploaded_image.getvalue()

                    base64_image = base64.b64encode(
                        image_bytes
                    ).decode("utf-8")

                    client_prompt = f"""
Eres SUPERZOO, un experto en animales.

Analiza cuidadosamente la imagen.

Pregunta del usuario:

{image_question}

Responde en español.

Diferencia claramente entre:
- lo que puedes observar directamente;
- y lo que solamente puedes inferir.

No inventes detalles que no puedan observarse.
Si no puedes identificar con seguridad el animal,
indica que la identificación es aproximada.
"""

                    # Cliente OpenAI moderno
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=api_key
                    )

                    result = client.responses.create(
                        model="gpt-5.6-luna",
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": client_prompt
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": (
                                            f"data:image/jpeg;base64,"
                                            f"{base64_image}"
                                        )
                                    }
                                ]
                            }
                        ]
                    )

                    image_answer = result.output_text

                    st.markdown("""
                    <div class="answer-card">

                    <h3>🦁 Análisis de SUPERZOO</h3>

                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(image_answer)

            except Exception as e:

                st.error(
                    f"❌ No se pudo analizar la imagen: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:20px; color:#166534;">

    🐾 <strong>SUPERZOO</strong> 🌿

    <br>

    Tu experto virtual en el mundo animal

    <br><br>

    🦁 🐘 🦜 🐍 🦋 🐼 🌱

    </div>
    """,
    unsafe_allow_html=True
)
