
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

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable
)


# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="EvaluaPro | Evaluaciones inteligentes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. DISEÑO VISUAL
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --verde-oscuro: #075B55;
    --verde-profundo: #034C47;
    --verde-medio: #0B796C;
    --verde-lima: #DDF18B;
    --verde-claro: #F4F7E9;
    --fondo: #F7F9F2;
    --texto: #173F3B;
    --gris: #71817B;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background-color: var(--fondo);
    color: var(--texto);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ENCABEZADO */

.hero {
    background: linear-gradient(
        125deg,
        #034C47 0%,
        #08736A 55%,
        #15947A 100%
    );
    padding: 42px;
    border-radius: 25px;
    color: white;
    margin-bottom: 32px;
    box-shadow: 0 12px 30px rgba(3, 76, 71, 0.12);
}

.hero h1 {
    font-size: 43px;
    font-weight: 800;
    color: white;
    letter-spacing: -1.5px;
    margin-bottom: 10px;
}

.hero p {
    color: #E5F5DB;
    font-size: 16px;
    line-height: 1.8;
    max-width: 680px;
}

.hero-tag {
    display: inline-block;
    background: #DDF18B;
    color: #075B55;
    padding: 9px 18px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-top: 12px;
}

/* TÍTULOS */

.section-title {
    color: #075B55;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin-top: 22px;
    margin-bottom: 12px;
}

/* TARJETAS */

.info-card {
    background: white;
    border: 1px solid #E5EBDD;
    border-radius: 19px;
    padding: 23px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(7,91,85,0.04);
}

.info-card h3 {
    color: #075B55;
    font-size: 17px;
    font-weight: 800;
}

.info-card p {
    color: #71817B;
    font-size: 13px;
    line-height: 1.6;
}

/* PREGUNTAS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 17px;
    border-color: #E2EBDD;
}

/* BOTONES */

div.stButton > button {
    background: #075B55;
    color: white;
    border: 1px solid #075B55;
    border-radius: 30px;
    padding: 12px 22px;
    min-height: 47px;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    background: #0C756A;
    color: #DDF18B;
    border-color: #0C756A;
    transform: translateY(-2px);
}

/* BOTONES DE DESCARGA */

div.stDownloadButton > button {
    background: #DDF18B;
    color: #075B55;
    border: 1px solid #DDF18B;
    border-radius: 30px;
    min-height: 47px;
    font-weight: 800;
    font-family: 'Manrope', sans-serif;
}

div.stDownloadButton > button:hover {
    background: #CBE879;
    color: #075B55;
    border-color: #CBE879;
}

/* BARRA LATERAL */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #034C47 0%,
        #075B55 65%,
        #08766A 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #DDF18B;
    font-weight: 800;
}

/* CAMPOS */

.stTextInput input,
.stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #DCE7D7;
    background: white;
}

.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #075B55;
}

/* CARGA DE ARCHIVOS */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #A6C68D;
    border-radius: 20px;
    padding: 20px;
}

/* MENSAJES */

.stAlert {
    border-radius: 14px;
}

/* MÉTRICAS */

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E5EBDD;
}

/* SEPARADORES */

hr {
    border-color: #DFE9D7;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. ENCABEZADO PRINCIPAL - CORREGIDO
# ==========================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(125deg, #034C47 0%, #08736A 55%, #15947A 100%);
        padding: 42px;
        border-radius: 25px;
        color: white;
        margin-bottom: 32px;
        box-shadow: 0 12px 30px rgba(3,76,71,0.12);
    ">
        <div style="
            color: #DDF18B;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 3px;
            margin-bottom: 15px;
        ">
            SMART EDUCATION PLATFORM
        </div>

        <h1 style="
            font-family: 'Manrope', sans-serif;
            font-size: 43px;
            font-weight: 800;
            color: white;
            margin-bottom: 10px;
        ">
            EvaluaPro.
        </h1>

        <p style="
            color: #E5F5DB;
            font-size: 16px;
            line-height: 1.8;
            max-width: 680px;
        ">
            Transforma tus materiales de clase en evaluaciones
            inteligentes. Sube tus documentos, personaliza
            tus preguntas y crea exámenes en segundos.
        </p>

        <div style="
            display: inline-block;
            background: #DDF18B;
            color: #075B55;
            padding: 9px 18px;
            border-radius: 30px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1px;
            margin-top: 12px;
        ">
            ✦ APRENDER · EVALUAR · MEJORAR
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# 4. CONFIGURACIÓN LATERAL
# ==========================================================

with st.sidebar:

    st.markdown("## ✦ EvaluaPro")

    st.caption("Plataforma inteligente para profesores")

    st.divider()

    st.markdown("### 🔑 Conexión")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="Introduce tu clave sk-...",
        help="Necesitas una clave API válida para generar evaluaciones."
    )

    st.caption(
        "La clave es necesaria para analizar documentos "
        "y generar preguntas. No la compartas ni la publiques."
    )

    if api_key:
        st.success("Clave introducida")
    else:
        st.warning("Falta introducir la clave API")

    st.divider()

    st.markdown("### 📊 Configuración del examen")

    numero_preguntas = st.slider(
        "Cantidad de preguntas",
        min_value=3,
        max_value=20,
        value=10
    )

    dificultad = st.selectbox(
        "Nivel de dificultad",
        ["Fácil", "Media", "Difícil"]
    )

    tipo = st.selectbox(
        "Tipo de evaluación",
        [
            "Opción múltiple",
            "Verdadero o falso",
            "Respuesta abierta"
        ]
    )

    st.divider()

    st.markdown("### 📚 ¿Cómo funciona?")

    st.markdown("""
    **01.** Introduce tu clave API.

    **02.** Sube el material de clase.

    **03.** Personaliza el examen.

    **04.** Genera las preguntas.

    **05.** Descarga los resultados.
    """)

    st.divider()

    st.markdown(
        "<div style='text-align:center;color:#DDF18B;"
        "font-size:11px;font-weight:700;'>"
        "EVALUAPRO · EDUCACIÓN INTELIGENTE"
        "</div>",
        unsafe_allow_html=True
    )


# ==========================================================
# 5. ESTADO DE LA APLICACIÓN
# ==========================================================

if "evaluacion" not in st.session_state:
    st.session_state.evaluacion = None

if "firma_evaluacion" not in st.session_state:
    st.session_state.firma_evaluacion = None

if "nombre_documento" not in st.session_state:
    st.session_state.nombre_documento = None

if "configuracion" not in st.session_state:
    st.session_state.configuracion = None


# ==========================================================
# 6. CARGA DEL DOCUMENTO
# ==========================================================

st.markdown(
    '<div class="section-title">📄 Carga tu material de clase</div>',
    unsafe_allow_html=True
)

st.write(
    "Sube un documento PDF con el contenido de tu clase, "
    "unidad o tema. EvaluaPro analizará su información "
    "para crear una evaluación personalizada."
)

pdf = st.file_uploader(
    "Selecciona o arrastra tu documento PDF",
    type=["pdf"],
    help="El documento debe contener texto seleccionable."
)


# ==========================================================
# 7. TARJETAS INFORMATIVAS
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="info-card">
        <div style="font-size:30px;">📄</div>
        <h3>Material de clase</h3>
        <p>
        Analiza tus documentos y utiliza
        su contenido como fuente de conocimiento.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="info-card">
        <div style="font-size:30px;">🧠</div>
        <h3>Inteligencia artificial</h3>
        <p>
        Genera preguntas educativas adaptadas
        al nivel de dificultad seleccionado.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="info-card">
        <div style="font-size:30px;">📥</div>
        <h3>Exámenes descargables</h3>
        <p>
        Descarga evaluaciones listas para
        utilizar con tus estudiantes.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 8. EXTRAER TEXTO DEL PDF
# ==========================================================

def extraer_texto(pdf_bytes):

    reader = PdfReader(io.BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError(
            "El PDF está protegido con contraseña. "
            "Sube una versión desbloqueada."
        )

    texto = "\n".join(
        pagina.extract_text() or ""
        for pagina in reader.pages
    )

    if not texto.strip():
        raise ValueError(
            "No se encontró texto en el documento. "
            "Puede ser un PDF escaneado sin OCR."
        )

    return texto


# ==========================================================
# 9. CREAR BASE DE CONOCIMIENTO RAG
# ==========================================================

def crear_base_conocimiento(texto, api_key):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    fragmentos = splitter.split_text(texto)

    if not fragmentos:
        raise ValueError(
            "No fue posible dividir el contenido del documento."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    base = FAISS.from_texts(
        fragmentos,
        embeddings
    )

    return base, len(fragmentos)


# ==========================================================
# 10. GENERAR EVALUACIÓN CON INTELIGENCIA ARTIFICIAL
# ==========================================================

def generar_evaluacion(
    base,
    api_key,
    numero_preguntas,
    dificultad,
    tipo
):

    cantidad_fragmentos = base.index.ntotal

    docs = base.similarity_search(
        "Conceptos principales, definiciones, características, "
        "explicaciones, ejemplos y datos importantes del documento.",
        k=min(15, cantidad_fragmentos)
    )

    contexto = "\n\n---\n\n".join(
        doc.page_content for doc in docs
    )

    if tipo == "Opción múltiple":

        formato = """
Cada pregunta debe tener cuatro opciones:
A, B, C y D.

Debe existir una sola respuesta correcta.
La respuesta debe ser una de las letras A, B, C o D.
"""

        ejemplo = """
{
  "numero": 1,
  "pregunta": "Texto de la pregunta",
  "opciones": {
    "A": "Primera opción",
    "B": "Segunda opción",
    "C": "Tercera opción",
    "D": "Cuarta opción"
  },
  "respuesta": "A",
  "explicacion": "Explicación de la respuesta correcta"
}
"""

    elif tipo == "Verdadero o falso":

        formato = """
Cada pregunta debe ser una afirmación que pueda
clasificarse como Verdadero o Falso.

La respuesta debe ser A para Verdadero o B para Falso.
"""

        ejemplo = """
{
  "numero": 1,
  "pregunta": "Texto de la afirmación",
  "opciones": {
    "A": "Verdadero",
    "B": "Falso"
  },
  "respuesta": "A",
  "explicacion": "Explicación de la respuesta"
}
"""

    else:

        formato = """
Cada pregunta debe ser abierta y permitir que el
estudiante responda con sus propias palabras.

Incluye una respuesta esperada y criterios básicos
para valorar la respuesta.
"""

        ejemplo = """
{
  "numero": 1,
  "pregunta": "Texto de la pregunta",
  "opciones": {},
  "respuesta": "Respuesta esperada",
  "explicacion": "Criterios para evaluar la respuesta"
}
"""

    instrucciones = f"""
Eres un experto en pedagogía y diseño de evaluaciones.

Tu tarea es crear un examen educativo a partir
del documento proporcionado.

REGLAS:

1. Genera exactamente {numero_preguntas} preguntas.
2. Nivel de dificultad: {dificultad}.
3. Tipo de evaluación: {tipo}.
4. Utiliza únicamente la información del documento.
5. No inventes datos ni conceptos.
6. Evita repetir preguntas.
7. Las preguntas deben ser claras y comprensibles.
8. Las respuestas deben estar respaldadas por el documento.
9. Responde exclusivamente con JSON válido.
10. No incluyas bloques Markdown.
11. No incluyas explicaciones fuera del JSON.

FORMATO:

{formato}

EJEMPLO DE UNA PREGUNTA:

{ejemplo}

ESTRUCTURA FINAL:

{{
  "titulo": "Evaluación de conocimientos",
  "preguntas": [
    ... preguntas aquí ...
  ]
}}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=api_key,
        max_retries=2
    )

    respuesta = llm.invoke([
        (
            "system",
            "Eres un generador profesional de evaluaciones "
            "educativas. Utiliza únicamente el contexto "
            "proporcionado y devuelve JSON válido."
        ),
        (
            "human",
            f"""
CONTENIDO DEL DOCUMENTO:

{contexto}

INSTRUCCIONES:

{instrucciones}
"""
        )
    ])

    contenido = respuesta.content

    if isinstance(contenido, list):
        contenido = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in contenido
        )

    contenido = contenido.strip()

    contenido = re.sub(
        r"^```(?:json)?\s*",
        "",
        contenido,
        flags=re.IGNORECASE
    )

    contenido = re.sub(
        r"\s*```$",
        "",
        contenido
    )

    resultado = json.loads(contenido)

    preguntas = resultado.get("preguntas", [])

    if not isinstance(preguntas, list) or not preguntas:
        raise ValueError(
            "La IA no devolvió preguntas válidas."
        )

    if len(preguntas) != numero_preguntas:
        raise ValueError(
            f"Se solicitaron {numero_preguntas} preguntas, "
            f"pero se generaron {len(preguntas)}. "
            "Intenta generar nuevamente."
        )

    for i, pregunta in enumerate(preguntas, start=1):

        if not isinstance(pregunta, dict):
            raise ValueError("Formato incorrecto de pregunta.")

        if not pregunta.get("pregunta"):
            raise ValueError("Hay una pregunta sin contenido.")

        if not pregunta.get("respuesta"):
            raise ValueError("Hay una pregunta sin respuesta.")

        opciones = pregunta.get("opciones", {})

        if not isinstance(opciones, dict):
            raise ValueError("Formato incorrecto de las opciones.")

        if tipo == "Opción múltiple":

            if not all(
                k in opciones for k in ["A", "B", "C", "D"]
            ):
                raise ValueError(
                    "Una pregunta no contiene cuatro opciones."
                )

            if pregunta["respuesta"] not in ["A", "B", "C", "D"]:
                raise ValueError(
                    "Una respuesta de opción múltiple es inválida."
                )

        elif tipo == "Verdadero o falso":

            if pregunta["respuesta"] not in ["A", "B"]:
                raise ValueError(
                    "Una respuesta de verdadero/falso es inválida."
                )

        pregunta["numero"] = i

    return {
        "titulo": resultado.get(
            "titulo",
            "Evaluación de conocimientos"
        ),
        "preguntas": preguntas
    }


# ==========================================================
# 11. CREAR DOCUMENTOS PDF
# ==========================================================

def crear_pdf(
    evaluacion,
    nombre_documento,
    incluir_respuestas=False
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48
    )

    estilos = getSampleStyleSheet()

    estilo_titulo = ParagraphStyle(
        "TituloPersonalizado",
        parent=estilos["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#075B55"),
        spaceAfter=10
    )

    estilo_subtitulo = ParagraphStyle(
        "Subtitulo",
        parent=estilos["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#71817B"),
        spaceAfter=20
    )

    estilo_pregunta = ParagraphStyle(
        "Pregunta",
        parent=estilos["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#075B55"),
        spaceBefore=12,
        spaceAfter=7
    )

    estilo_texto = ParagraphStyle(
        "Texto",
        parent=estilos["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=5
    )

    estilo_respuesta = ParagraphStyle(
        "Respuesta",
        parent=estilos["Normal"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#166534"),
        spaceAfter=5
    )

    elementos = []

    titulo = (
        "CLAVE DE RESPUESTAS"
        if incluir_respuestas
        else "EVALUACIÓN DE CONOCIMIENTOS"
    )

    elementos.append(
        Paragraph(titulo, estilo_titulo)
    )

    elementos.append(
        Paragraph(
            f"Material de referencia: "
            f"{html.escape(nombre_documento)}",
            estilo_subtitulo
        )
    )

    elementos.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#DCE7D7")
        )
    )

    elementos.append(Spacer(1, 20))

    if not incluir_respuestas:

        elementos.append(
            Paragraph(
                "Nombre del estudiante: "
                "________________________________",
                estilo_texto
            )
        )

        elementos.append(
            Paragraph(
                "Fecha: _____________________ "
                "Curso: _____________________",
                estilo_texto
            )
        )

        elementos.append(Spacer(1, 15))

    for pregunta in evaluacion["preguntas"]:

        numero = pregunta["numero"]

        texto_pregunta = html.escape(
            str(pregunta["pregunta"])
        )

        elementos.append(
            Paragraph(
                f"{numero}. {texto_pregunta}",
                estilo_pregunta
            )
        )

        opciones = pregunta.get("opciones", {})

        if opciones:

            for letra, opcion in opciones.items():

                elementos.append(
                    Paragraph(
                        f"<b>{html.escape(str(letra))}.</b> "
                        f"{html.escape(str(opcion))}",
                        estilo_texto
                    )
                )

        elif not incluir_respuestas:

            elementos.append(
                Paragraph(
                    "Respuesta: ______________________________",
                    estilo_texto
                )
            )

            elementos.append(
                Paragraph(
                    "________________________________________",
                    estilo_texto
                )
            )

        if incluir_respuestas:

            respuesta = html.escape(
                str(pregunta["respuesta"])
            )

            explicacion = html.escape(
                str(pregunta.get("explicacion", ""))
            )

            elementos.append(
                Paragraph(
                    f"<b>Respuesta correcta / esperada:</b> "
                    f"{respuesta}",
                    estilo_respuesta
                )
            )

            if explicacion:

                elementos.append(
                    Paragraph(
                        f"<b>Explicación:</b> {explicacion}",
                        estilo_respuesta
                    )
                )

        elementos.append(Spacer(1, 10))

    doc.build(elementos)

    buffer.seek(0)

    return buffer.getvalue()


# ==========================================================
# 12. VALIDAR DOCUMENTO Y CONFIGURACIÓN
# ==========================================================

if pdf is not None:

    firma_actual = hashlib.sha256(
        pdf.getvalue()
        + str(numero_preguntas).encode()
        + dificultad.encode()
        + tipo.encode()
    ).hexdigest()

    if firma_actual != st.session_state.firma_evaluacion:

        st.session_state.evaluacion = None
        st.session_state.nombre_documento = None
        st.session_state.configuracion = None

else:

    firma_actual = None

    st.session_state.evaluacion = None
    st.session_state.firma_evaluacion = None
    st.session_state.nombre_documento = None
    st.session_state.configuracion = None


# ==========================================================
# 13. INFORMACIÓN DEL DOCUMENTO
# ==========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📋 Tu material de trabajo</div>',
    unsafe_allow_html=True
)

if pdf is None:

    st.info(
        "📄 Selecciona un documento PDF para comenzar."
    )

else:

    st.success(
        f"Documento seleccionado: {pdf.name}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tamaño del documento",
            f"{pdf.size / 1024:.1f} KB"
        )

    with col2:
        st.metric(
            "Preguntas configuradas",
            numero_preguntas
        )


# ==========================================================
# 14. GENERACIÓN DE LA EVALUACIÓN
# ==========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">✨ Crea tu evaluación</div>',
    unsafe_allow_html=True
)

st.write(
    "Cuando hayas introducido tu clave API y seleccionado "
    "el documento, podrás generar tu examen."
)

if not api_key:

    st.warning(
        "🔑 Debes introducir tu clave API de OpenAI "
        "en el menú lateral para continuar."
    )

elif pdf is None:

    st.info(
        "📄 Primero debes cargar un documento PDF."
    )

else:

    if st.button(
        "✦ GENERAR EVALUACIÓN",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "📖 Extrayendo información del documento..."
            ):

                texto = extraer_texto(pdf.getvalue())

            with st.spinner(
                "🧠 Creando la base de conocimiento..."
            ):

                base, fragmentos = crear_base_conocimiento(
                    texto,
                    api_key
                )

            st.info(
                f"Documento analizado: {len(texto):,} caracteres "
                f"y {fragmentos} fragmentos."
            )

            with st.spinner(
                "✍️ La inteligencia artificial está creando "
                "tu evaluación..."
            ):

                resultado = generar_evaluacion(
                    base,
                    api_key,
                    numero_preguntas,
                    dificultad,
                    tipo
                )

            st.session_state.evaluacion = resultado

            st.session_state.nombre_documento = pdf.name

            st.session_state.configuracion = {
                "cantidad": numero_preguntas,
                "dificultad": dificultad,
                "tipo": tipo
            }

            st.session_state.firma_evaluacion = firma_actual

            st.success(
                "¡Tu evaluación se ha generado correctamente!"
            )

        except Exception as error:

            st.error(
                "No fue posible generar la evaluación."
            )

            st.exception(error)


# ==========================================================
# 15. MOSTRAR EVALUACIÓN
# ==========================================================

if (
    st.session_state.evaluacion
    and st.session_state.firma_evaluacion == firma_actual
):

    evaluacion = st.session_state.evaluacion

    configuracion = st.session_state.configuracion

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📋 Tu evaluación está lista</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Preguntas",
        len(evaluacion["preguntas"])
    )

    col2.metric(
        "Dificultad",
        configuracion["dificultad"]
    )

    col3.metric(
        "Modalidad",
        configuracion["tipo"]
    )

    st.markdown("### 📝 Vista previa del examen")

    st.caption(
        "Revisa las preguntas antes de descargar el documento."
    )

    for pregunta in evaluacion["preguntas"]:

        with st.container(border=True):

            st.markdown(
                f"**Pregunta {pregunta['numero']}**"
            )

            st.write(pregunta["pregunta"])

            opciones = pregunta.get("opciones", {})

            if opciones:

                for letra, opcion in opciones.items():

                    st.write(
                        f"**{letra}.** {opcion}"
                    )

            else:

                st.markdown(
                    "_Respuesta abierta_"
                )

                st.text_area(
                    "Espacio para responder",
                    key=f"respuesta_{pregunta['numero']}",
                    disabled=True
                )

    # ======================================================
    # 16. DESCARGAR DOCUMENTOS
    # ======================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📥 Descarga tus documentos</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Descarga el examen para tus estudiantes y la clave "
        "de respuestas para uso exclusivo del profesor."
    )

    examen_pdf = crear_pdf(
        evaluacion,
        st.session_state.nombre_documento,
        incluir_respuestas=False
    )

    respuestas_pdf = crear_pdf(
        evaluacion,
        st.session_state.nombre_documento,
        incluir_respuestas=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="📄 Descargar examen PDF",
            data=examen_pdf,
            file_name="Examen_EvaluaPro.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col2:

        st.download_button(
            label="🔑 Descargar respuestas PDF",
            data=respuestas_pdf,
            file_name="Respuestas_EvaluaPro.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # ======================================================
    # 17. RESPUESTAS DEL PROFESOR
    # ======================================================

    with st.expander("🔍 Consultar clave de respuestas"):

        for pregunta in evaluacion["preguntas"]:

            st.markdown(
                f"**Pregunta {pregunta['numero']}**"
            )

            st.write(
                f"Respuesta: {pregunta['respuesta']}"
            )

            st.write(
                f"Explicación: "
                f"{pregunta.get('explicacion', '')}"
            )

            st.divider()

    # ======================================================
    # 18. NUEVA EVALUACIÓN
    # ======================================================

    if st.button(
        "🔄 Crear una nueva evaluación",
        use_container_width=True
    ):

        st.session_state.evaluacion = None
        st.session_state.firma_evaluacion = None
        st.session_state.nombre_documento = None
        st.session_state.configuracion = None

        st.rerun()


# ==========================================================
# 19. PIE DE PÁGINA
# ==========================================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    padding:25px 10px;
    color:#71817B;
">

    <div style="
        font-size:19px;
        font-weight:800;
        color:#075B55;
        margin-bottom:8px;
    ">
        EvaluaPro.
    </div>

    <div style="font-size:12px;">
        Aprende · Evalúa · Mejora
    </div>

    <div style="
        font-size:11px;
        margin-top:12px;
    ">
        Plataforma inteligente para la creación de evaluaciones educativas.
    </div>

</div>
""", unsafe_allow_html=True)
