
import io
import json
import re
import html
import hashlib
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    KeepTogether
)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="EvaluaPro",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ESTILOS CSS
# =========================================================

st.markdown("""
<style>

/* -----------------------------------------------------
   FUENTE Y FONDO GENERAL
----------------------------------------------------- */

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background: #F7F9F2;
}

[data-testid="stAppViewContainer"] {
    background: #F7F9F2;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* -----------------------------------------------------
   BARRA LATERAL
----------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #034C47 0%,
        #075B55 55%,
        #034C47 100%
    );
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF;
}

/* Inputs de la barra lateral */

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background-color: #FFFFFF !important;
    color: #202020 !important;
    border-radius: 9px !important;
    border: 1px solid #D5DDD7 !important;
}

/* Texto escrito */

section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: #777777 !important;
    opacity: 1 !important;
}

/* Selectores */

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #202020 !important;
    border-radius: 9px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #202020 !important;
}

/* Slider */

section[data-testid="stSidebar"] [data-testid="stSlider"] {
    color: #DDF18B;
}

/* Mensajes de ayuda */

section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
    color: #DDF18B !important;
}


/* -----------------------------------------------------
   CORRECCIÓN BOTÓN MOSTRAR/OCULTAR API KEY
----------------------------------------------------- */

/* Oculta únicamente el botón de visibilidad del campo password */

section[data-testid="stSidebar"]
div[data-testid="stTextInput"]:has(input[type="password"]) button {
    display: none !important;
}


/* -----------------------------------------------------
   CORRECCIÓN BOTÓN DE CONTRAER / EXPANDIR SIDEBAR
----------------------------------------------------- */

/* Botón para cerrar la barra lateral cuando está abierta */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: #DDF18B !important;
    border: 1px solid #DDF18B !important;
    border-radius: 9px !important;
    width: 38px !important;
    height: 38px !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

/* Botón para abrir la barra lateral cuando está cerrada */

button[data-testid="stSidebarExpandButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: #DDF18B !important;
    border: 1px solid #DDF18B !important;
    border-radius: 9px !important;
    width: 38px !important;
    height: 38px !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
}

/* Mantener visibles los iconos SVG */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="stSidebarExpandButton"] svg {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 20px !important;
    height: 20px !important;
    color: #034C47 !important;
    stroke: #034C47 !important;
}

/* No ocultar los elementos internos que contienen el icono */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"] span,
button[data-testid="stSidebarExpandButton"] span {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Hover de los botones */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="stSidebarExpandButton"]:hover {
    background-color: #FFFFFF !important;
    border-color: #DDF18B !important;
}


/* -----------------------------------------------------
   SCROLLBAR DE LA BARRA LATERAL
----------------------------------------------------- */

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    scrollbar-width: thin;
    scrollbar-color: #DDF18B #034C47;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
    width: 9px;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
    background: #034C47;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
    background: #DDF18B;
    border-radius: 10px;
    border: 2px solid #034C47;
}


/* -----------------------------------------------------
   CONTENIDO PRINCIPAL
----------------------------------------------------- */

.main .block-container {
    color: #034C47;
}

.main h1,
.main h2,
.main h3,
.main p,
.main label {
    color: #034C47;
}


/* -----------------------------------------------------
   BOTONES
----------------------------------------------------- */

.stButton > button {
    background: linear-gradient(
        110deg,
        #034C47,
        #08736A
    ) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.3rem !important;
    font-weight: 700 !important;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #08736A !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(3, 76, 71, 0.18);
}


/* -----------------------------------------------------
   CARGADOR DE ARCHIVOS
----------------------------------------------------- */

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #A7BBAE;
    border-radius: 14px;
    padding: 1rem;
}

[data-testid="stFileUploader"] * {
    color: #034C47 !important;
}


/* -----------------------------------------------------
   MÉTRICAS Y CONTENEDORES
----------------------------------------------------- */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E9DF;
    border-radius: 12px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    color: #55736A !important;
}

[data-testid="stMetricValue"] {
    color: #034C47 !important;
}


/* -----------------------------------------------------
   EXPANDERS
----------------------------------------------------- */

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E9DF;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #034C47 !important;
}


/* -----------------------------------------------------
   ALERTAS
----------------------------------------------------- */

[data-testid="stAlert"] {
    border-radius: 10px;
}


/* -----------------------------------------------------
   DESCARGAS
----------------------------------------------------- */

.stDownloadButton > button {
    background: #DDF18B !important;
    color: #034C47 !important;
    border: 1px solid #DDF18B !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
}

.stDownloadButton > button:hover {
    background: #CBE76C !important;
    color: #034C47 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# ENCABEZADO PRINCIPAL
# =========================================================

st.html("""
<div style="
    background: linear-gradient(125deg, #034C47 0%, #08736A 55%, #15947A 100%);
    padding: 38px 42px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 24px rgba(3, 76, 71, 0.13);
">
    <div style="
        font-size: 11px;
        letter-spacing: 3px;
        font-weight: 800;
        color: #DDF18B;
        margin-bottom: 12px;
    ">
        SMART EDUCATION PLATFORM
    </div>

    <h1 style="
        font-family: Manrope, sans-serif;
        color: white;
        font-size: 48px;
        font-weight: 800;
        margin: 0 0 12px 0;
    ">
        EvaluaPro.
    </h1>

    <p style="
        color: #FFFFFF;
        font-size: 16px;
        line-height: 1.7;
        max-width: 750px;
        margin: 0 0 22px 0;
    ">
        Transforma tus materiales de clase en evaluaciones inteligentes.
        Sube tus documentos, personaliza tus preguntas y crea exámenes en segundos.
    </p>

    <div style="
        display: inline-block;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 9px 15px;
        border-radius: 30px;
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 800;
        color: #DDF18B;
    ">
        ✦ APRENDER · EVALUAR · MEJORAR
    </div>
</div>
""")


# =========================================================
# BARRA LATERAL: CONFIGURACIÓN
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Configuración")

    st.markdown(
        "Personaliza tu evaluación antes de comenzar."
    )

    st.markdown("---")

    st.markdown("### 🔐 Conexión con OpenAI")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="Introduce tu clave sk-...",
        help="Tu clave se utiliza para conectar con OpenAI durante la generación."
    ).strip()

    if api_key:
        st.success("✓ Clave API introducida")

        st.html("""
        <div style="
            background: rgba(221,241,139,0.15);
            border: 1px solid #DDF18B;
            border-radius: 10px;
            padding: 12px;
            color: white;
            font-size: 12px;
            line-height: 1.6;
        ">
            ✓ Ya puedes cargar tu documento PDF y generar tu evaluación.
        </div>
        """)
    else:
        st.warning("🔑 Introduce tu clave API para comenzar.")

    st.markdown("---")

    st.markdown("### 📝 Parámetros del examen")

    question_count = st.slider(
        "Número de preguntas",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )

    difficulty = st.selectbox(
        "Nivel de dificultad",
        ["Fácil", "Media", "Difícil"],
        index=1
    )

    question_type = st.selectbox(
        "Tipo de preguntas",
        [
            "Opción múltiple",
            "Verdadero o falso",
            "Respuesta abierta"
        ]
    )

    st.markdown("---")

    st.caption(
        "EvaluaPro · SMART EDUCATION PLATFORM"
    )


# =========================================================
# ESTADO DE LA APLICACIÓN
# =========================================================

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "exam_pdf" not in st.session_state:
    st.session_state.exam_pdf = None

if "answer_pdf" not in st.session_state:
    st.session_state.answer_pdf = None

if "signature" not in st.session_state:
    st.session_state.signature = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "config_signature" not in st.session_state:
    st.session_state.config_signature = None


# =========================================================
# FUNCIONES: EXTRACCIÓN DE PDF
# =========================================================

def extract_pdf_text(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(uploaded_file)

    if reader.is_encrypted:
        raise ValueError(
            "El PDF está protegido con contraseña. "
            "Carga un documento sin protección."
        )

    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages_text.append(text)

    full_text = "\n\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "No se encontró texto en el PDF. "
            "Si el documento está escaneado, primero debes aplicar OCR."
        )

    return full_text, len(reader.pages)


# =========================================================
# FUNCIONES: DIVISIÓN DEL TEXTO
# =========================================================

def split_document(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError(
            "No fue posible dividir el contenido del documento."
        )

    return chunks


# =========================================================
# FUNCIONES: GENERACIÓN DE PREGUNTAS
# =========================================================

def generate_questions(
    vector_store,
    api_key,
    question_count,
    difficulty,
    question_type
):

    retriever_results = vector_store.similarity_search(
        "Conceptos principales, definiciones, explicaciones, "
        "datos importantes y conocimientos fundamentales del documento.",
        k=15
    )

    context = "\n\n".join(
        document.page_content
        for document in retriever_results
    )

    if question_type == "Opción múltiple":

        format_instructions = """
Cada pregunta debe tener exactamente cuatro opciones:
A, B, C y D.

La respuesta correcta debe ser una sola letra:
A, B, C o D.
"""

        example = """
{
  "questions": [
    {
      "question": "¿Cuál es la idea principal del texto?",
      "options": {
        "A": "Primera opción",
        "B": "Segunda opción",
        "C": "Tercera opción",
        "D": "Cuarta opción"
      },
      "answer": "B",
      "explanation": "Explicación breve de la respuesta."
    }
  ]
}
"""

    elif question_type == "Verdadero o falso":

        format_instructions = """
Cada pregunta debe ser una afirmación que pueda ser verdadera o falsa.

Usa estas opciones:
A: Verdadero
B: Falso

La respuesta correcta debe ser A o B.
"""

        example = """
{
  "questions": [
    {
      "question": "El concepto explicado en el documento es correcto.",
      "options": {
        "A": "Verdadero",
        "B": "Falso"
      },
      "answer": "A",
      "explanation": "Explicación breve de la respuesta."
    }
  ]
}
"""

    else:

        format_instructions = """
Cada pregunta debe ser abierta y permitir una respuesta desarrollada.

Incluye una respuesta esperada que sirva como guía de corrección.
No incluyas opciones.
"""

        example = """
{
  "questions": [
    {
      "question": "Explica los principales conceptos presentados en el documento.",
      "answer": "Respuesta esperada con los elementos fundamentales.",
      "explanation": "Criterio para valorar la respuesta."
    }
  ]
}
"""

    prompt = f"""
Eres un experto en diseño de evaluaciones educativas.

Debes crear una evaluación basada exclusivamente en el contenido
del documento proporcionado.

CARACTERÍSTICAS:

- Número exacto de preguntas: {question_count}
- Dificultad: {difficulty}
- Tipo: {question_type}
- Idioma: español.
- Evita preguntas repetidas.
- Evita información que no aparezca en el documento.
- Formula preguntas claras y académicas.
- Distribuye las preguntas entre los temas disponibles.
- No inventes datos ni conceptos.

INSTRUCCIONES DEL FORMATO:

{format_instructions}

Devuelve exclusivamente un JSON válido con esta estructura:

{example}

CONTENIDO DEL DOCUMENTO:

{context}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    content = str(content).strip()

    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    try:
        result = json.loads(content)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "La IA no devolvió un JSON válido."
            )

        result = json.loads(match.group(0))

    questions = result.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "La respuesta generada no contiene una lista de preguntas."
        )

    if len(questions) != question_count:
        raise ValueError(
            f"La IA generó {len(questions)} preguntas "
            f"en lugar de las {question_count} solicitadas. "
            "Inténtalo nuevamente."
        )

    for index, question in enumerate(questions, start=1):

        if not isinstance(question, dict):
            raise ValueError(
                f"La pregunta {index} tiene un formato incorrecto."
            )

        if not question.get("question"):
            raise ValueError(
                f"La pregunta {index} no tiene enunciado."
            )

        if not question.get("answer"):
            raise ValueError(
                f"La pregunta {index} no tiene respuesta."
            )

        if question_type in [
            "Opción múltiple",
            "Verdadero o falso"
        ]:

            options = question.get("options")

            if not isinstance(options, dict):
                raise ValueError(
                    f"La pregunta {index} no contiene opciones válidas."
                )

            expected_keys = (
                ["A", "B", "C", "D"]
                if question_type == "Opción múltiple"
                else ["A", "B"]
            )

            for key in expected_keys:
                if not options.get(key):
                    raise ValueError(
                        f"La pregunta {index} no tiene la opción {key}."
                    )

            if question["answer"] not in expected_keys:
                raise ValueError(
                    f"La respuesta de la pregunta {index} "
                    "no corresponde a una opción válida."
                )

    return questions


# =========================================================
# FUNCIONES: GENERACIÓN DE PDF
# =========================================================

def make_pdf(title, questions, question_type, answer_key=False):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="EvalTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#034C47"),
            spaceAfter=10
        )
    )

    styles.add(
        ParagraphStyle(
            name="EvalSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#55736A"),
            spaceAfter=22
        )
    )

    styles.add(
        ParagraphStyle(
            name="QuestionStyle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#034C47"),
            spaceBefore=12,
            spaceAfter=7
        )
    )

    styles.add(
        ParagraphStyle(
            name="OptionStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            leftIndent=14,
            textColor=colors.HexColor("#333333"),
            spaceAfter=3
        )
    )

    styles.add(
        ParagraphStyle(
            name="AnswerStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#333333"),
            leftIndent=10,
            spaceAfter=5
        )
    )

    story = []

    if answer_key:
        title = "Solucionario · " + title

    story.append(
        Paragraph(
            html.escape(title),
            styles["EvalTitle"]
        )
    )

    subtitle = (
        "EvaluaPro · SMART EDUCATION PLATFORM"
        if not answer_key
        else "Respuestas y criterios de corrección"
    )

    story.append(
        Paragraph(
            subtitle,
            styles["EvalSubtitle"]
        )
    )

    if not answer_key:

        story.append(
            Paragraph(
                "<b>Nombre:</b> __________________________________________",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 10))

        story.append(
            Paragraph(
                "<b>Fecha:</b> ____________________ "
                "&nbsp;&nbsp;&nbsp;&nbsp; "
                "<b>Curso:</b> ____________________",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 18))

    else:

        story.append(
            Paragraph(
                "Documento de respuestas para el docente.",
                styles["AnswerStyle"]
            )
        )

        story.append(Spacer(1, 12))

    for index, question in enumerate(questions, start=1):

        question_text = html.escape(
            str(question.get("question", ""))
        )

        if answer_key:

            answer = str(question.get("answer", ""))
            explanation = str(
                question.get("explanation", "")
            )

            if question_type in [
                "Opción múltiple",
                "Verdadero o falso"
            ]:

                options = question.get("options", {})
                answer_text = options.get(answer, answer)

                story.append(
                    Paragraph(
                        f"<b>{index}. {question_text}</b>",
                        styles["QuestionStyle"]
                    )
                )

                story.append(
                    Paragraph(
                        f"<b>Respuesta:</b> "
                        f"{html.escape(str(answer))} — "
                        f"{html.escape(str(answer_text))}",
                        styles["AnswerStyle"]
                    )
                )

            else:

                story.append(
                    Paragraph(
                        f"<b>{index}. {question_text}</b>",
                        styles["QuestionStyle"]
                    )
                )

                story.append(
                    Paragraph(
                        f"<b>Respuesta esperada:</b> "
                        f"{html.escape(answer)}",
                        styles["AnswerStyle"]
                    )
                )

            if explanation:
                story.append(
                    Paragraph(
                        f"<b>Criterio / explicación:</b> "
                        f"{html.escape(explanation)}",
                        styles["AnswerStyle"]
                    )
                )

        else:

            story.append(
                Paragraph(
                    f"{index}. {question_text}",
                    styles["QuestionStyle"]
                )
            )

            options = question.get("options", {})

            if question_type in [
                "Opción múltiple",
                "Verdadero o falso"
            ]:

                for key, value in options.items():

                    story.append(
                        Paragraph(
                            f"{html.escape(str(key))}. "
                            f"{html.escape(str(value))}",
                            styles["OptionStyle"]
                        )
                    )

            else:

                story.append(Spacer(1, 12))

                story.append(
                    Paragraph(
                        "________________________________________________________________________",
                        styles["OptionStyle"]
                    )
                )

                story.append(
                    Paragraph(
                        "________________________________________________________________________",
                        styles["OptionStyle"]
                    )
                )

                story.append(
                    Paragraph(
                        "________________________________________________________________________",
                        styles["OptionStyle"]
                    )
                )

    def add_page_number(canvas, document):

        canvas.saveState()

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(
            colors.HexColor("#55736A")
        )

        canvas.drawString(
            2 * cm,
            1.1 * cm,
            "EvaluaPro · SMART EDUCATION PLATFORM"
        )

        canvas.drawRightString(
            A4[0] - 2 * cm,
            1.1 * cm,
            f"Página {document.page}"
        )

        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# ÁREA PRINCIPAL: CARGA DEL DOCUMENTO
# =========================================================

st.markdown(
    "## 📄 Carga tu documento"
)

st.markdown(
    "Sube el material de estudio en PDF para convertirlo "
    "en una evaluación."
)

uploaded_file = st.file_uploader(
    "Selecciona un archivo PDF",
    type=["pdf"],
    help="Carga un documento PDF que contenga texto seleccionable."
)


# =========================================================
# VALIDACIÓN DE CONFIGURACIÓN Y DOCUMENTO
# =========================================================

current_config = (
    question_count,
    difficulty,
    question_type
)

if uploaded_file:

    document_signature = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    current_signature = (
        document_signature,
        current_config
    )

    if st.session_state.signature != current_signature:

        st.session_state.evaluation = None
        st.session_state.exam_pdf = None
        st.session_state.answer_pdf = None
        st.session_state.signature = current_signature

    st.success(
        f"Documento cargado: {uploaded_file.name}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documento",
            "PDF"
        )

    with col2:
        st.metric(
            "Preguntas configuradas",
            question_count
        )

else:

    st.info(
        "Carga un documento PDF para comenzar."
    )


# =========================================================
# GENERAR EVALUACIÓN
# =========================================================

st.markdown("---")

st.markdown(
    "## ✨ Generar evaluación"
)

st.markdown(
    "EvaluaPro analizará el documento y creará preguntas "
    "según los parámetros seleccionados."
)

can_generate = bool(
    uploaded_file and api_key
)

if not api_key:
    st.warning(
        "Primero introduce tu clave API de OpenAI en la barra lateral."
    )

if uploaded_file and api_key:

    if st.button(
        "✨ Generar evaluación",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Leyendo y analizando el documento PDF..."
            ):

                document_text, page_count = extract_pdf_text(
                    uploaded_file
                )

            with st.spinner(
                "Preparando el contenido para el análisis..."
            ):

                chunks = split_document(
                    document_text
                )

            with st.spinner(
                "Creando la base de conocimiento..."
            ):

                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=api_key
                )

                vector_store = FAISS.from_texts(
                    chunks,
                    embeddings
                )

            with st.spinner(
                "Diseñando las preguntas de tu evaluación..."
            ):

                questions = generate_questions(
                    vector_store=vector_store,
                    api_key=api_key,
                    question_count=question_count,
                    difficulty=difficulty,
                    question_type=question_type
                )

            with st.spinner(
                "Preparando los documentos PDF..."
            ):

                exam_pdf = make_pdf(
                    title="Evaluación académica",
                    questions=questions,
                    question_type=question_type,
                    answer_key=False
                )

                answer_pdf = make_pdf(
                    title="Evaluación académica",
                    questions=questions,
                    question_type=question_type,
                    answer_key=True
                )

            st.session_state.evaluation = questions
            st.session_state.exam_pdf = exam_pdf
            st.session_state.answer_pdf = answer_pdf
            st.session_state.document_name = uploaded_file.name
            st.session_state.config_signature = current_config

            st.success(
                f"¡Evaluación generada correctamente! "
                f"Se crearon {len(questions)} preguntas "
                f"a partir de {page_count} páginas."
            )

        except Exception as error:

            st.error(
                "No fue posible generar la evaluación."
            )

            st.exception(error)


# =========================================================
# VISTA PREVIA Y DESCARGAS
# =========================================================

if st.session_state.evaluation:

    st.markdown("---")

    st.markdown(
        "## 📚 Tu evaluación está lista"
    )

    st.markdown(
        "Revisa las preguntas y descarga los documentos."
    )

    questions = st.session_state.evaluation

    for index, question in enumerate(
        questions,
        start=1
    ):

        with st.container(border=True):

            st.markdown(
                f"**Pregunta {index}.** "
                f"{question['question']}"
            )

            if question_type in [
                "Opción múltiple",
                "Verdadero o falso"
            ]:

                options = question.get("options", {})

                for key, value in options.items():

                    st.markdown(
                        f"- **{key}.** {value}"
                    )

            else:

                st.text_area(
                    "Espacio para responder",
                    key=f"answer_space_{index}",
                    disabled=True,
                    placeholder="Escribe aquí tu respuesta...",
                    height=90
                )

    st.markdown("---")

    st.markdown(
        "## ⬇️ Descargar documentos"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="📄 Descargar examen PDF",
            data=st.session_state.exam_pdf,
            file_name="EvaluaPro_Examen.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="🔑 Descargar solucionario PDF",
            data=st.session_state.answer_pdf,
            file_name="EvaluaPro_Solucionario.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with st.expander(
        "👁️ Ver respuestas correctas"
    ):

        for index, question in enumerate(
            questions,
            start=1
        ):

            answer = question.get("answer", "")

            if question_type in [
                "Opción múltiple",
                "Verdadero o falso"
            ]:

                options = question.get("options", {})
                answer_text = options.get(answer, answer)

                st.markdown(
                    f"**Pregunta {index}:** "
                    f"{answer} — {answer_text}"
                )

            else:

                st.markdown(
                    f"**Pregunta {index}:** {answer}"
                )

            explanation = question.get(
                "explanation",
                ""
            )

            if explanation:
                st.caption(explanation)

    st.markdown("---")

    if st.button(
        "🔄 Crear nueva evaluación",
        use_container_width=True
    ):

        st.session_state.evaluation = None
        st.session_state.exam_pdf = None
        st.session_state.answer_pdf = None
        st.session_state.signature = None
        st.session_state.document_name = None

        st.rerun()


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.markdown("---")

st.html("""
<div style="
    text-align: center;
    padding: 18px 10px;
    color: #55736A;
    font-family: Manrope, sans-serif;
    font-size: 12px;
">
    <div style="
        color: #034C47;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    ">
        EvaluaPro.
    </div>

    SMART EDUCATION PLATFORM

    <div style="margin-top: 8px;">
        APRENDER · EVALUAR · MEJORAR
    </div>
</div>
""")
