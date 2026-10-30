
import io
import json
import re
import html
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
# CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="EvaluaPro | Generador de evaluaciones",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# DISEÑO VISUAL
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f6fb;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e40af);
    padding: 38px 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 17px;
}

.section-title {
    color: #172554;
    font-size: 23px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.question-card {
    background: white;
    border: 1px solid #dce4f0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.question-title {
    color: #1e3a8a;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 12px;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 45px;
}

div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 45px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>📝 EvaluaPro</h1>
    <p>
        Generador inteligente de evaluaciones educativas.
        Convierte tus documentos de clase en exámenes
        de forma rápida y sencilla.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# BARRA LATERAL
# ==========================================================

with st.sidebar:

    st.markdown("## ⚙️ Configuración")

    st.markdown(
        "Personaliza tu evaluación antes de generarla."
    )

    st.divider()

    st.markdown("### 🔑 Conexión con OpenAI")

    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        placeholder="Introduce tu clave sk-...",
        help="Necesitas una clave API válida para generar evaluaciones."
    )

    st.caption(
        "La clave se utiliza para conectar con los servicios "
        "de OpenAI. No la publiques en GitHub."
    )

    st.divider()

    st.markdown("### 📊 Cantidad de preguntas")

    numero_preguntas = st.slider(
        "Selecciona la cantidad",
        min_value=3,
        max_value=20,
        value=10
    )

    st.markdown("### 🎯 Nivel de dificultad")

    dificultad = st.selectbox(
        "Dificultad",
        ["Fácil", "Media", "Difícil"]
    )

    st.markdown("### 📚 Tipo de evaluación")

    tipo = st.selectbox(
        "Selecciona el formato",
        [
            "Opción múltiple",
            "Verdadero o falso",
            "Respuesta abierta"
        ]
    )

    st.divider()

    st.markdown("""
    **¿Cómo funciona?**

    1. Introduce tu clave API.
    2. Sube un documento PDF.
    3. Configura tu evaluación.
    4. Genera las preguntas.
    5. Descarga el examen.
    """)

    st.caption("EvaluaPro · Herramienta educativa")


# ==========================================================
# ESTADO DE LA APLICACIÓN
# ==========================================================

if "evaluacion" not in st.session_state:
    st.session_state.evaluacion = None

if "nombre_documento" not in st.session_state:
    st.session_state.nombre_documento = None

if "configuracion" not in st.session_state:
    st.session_state.configuracion = None


# ==========================================================
# INFORMACIÓN PRINCIPAL
# ==========================================================

st.markdown(
    '<div class="section-title">📄 Carga tu material de clase</div>',
    unsafe_allow_html=True
)

st.write(
    "Sube un documento PDF con el contenido de tu clase, "
    "unidad o tema. EvaluaPro analizará su información "
    "para crear una evaluación."
)

pdf = st.file_uploader(
    "Selecciona o arrastra tu documento PDF",
    type=["pdf"],
    help="El documento debe contener texto seleccionable."
)


# ==========================================================
# INFORMACIÓN DE CONFIGURACIÓN
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📄 Documento</h3>
        <p>Analiza tus materiales de clase.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Inteligencia artificial</h3>
        <p>Genera preguntas según el contenido.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>📥 Descarga</h3>
        <p>Obtén tu evaluación en PDF.</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# PROCESAMIENTO DEL DOCUMENTO
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
            "Es posible que sea un PDF escaneado. "
            "Necesitas un documento con texto seleccionable."
        )

    return texto


# ==========================================================
# CREAR BASE DE CONOCIMIENTO RAG
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
# GENERAR EVALUACIÓN CON IA
# ==========================================================

def generar_evaluacion(
    base,
    api_key,
    numero_preguntas,
    dificultad,
    tipo
):

    cantidad_fragmentos = base.index.ntotal

    # Recuperar fragmentos relevantes del documento.
    docs = base.similarity_search(
        "Temas principales, conceptos, definiciones, "
        "explicaciones, características, ejemplos y "
        "datos importantes del documento.",
        k=min(15, cantidad_fragmentos)
    )

    contexto = "\n\n".join(
        doc.page_content for doc in docs
    )

    if tipo == "Opción múltiple":

        formato = """
Cada pregunta debe tener cuatro opciones:
A, B, C y D.

Debe existir una sola respuesta correcta.
"""

        estructura = """
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
"""

        estructura = """
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
Cada pregunta debe ser abierta y permitir que
el estudiante responda con sus propias palabras.
"""

        estructura = """
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

Tu tarea es crear un examen educativo a partir del
documento proporcionado.

REGLAS:

1. Genera exactamente {numero_preguntas} preguntas.
2. Nivel de dificultad: {dificultad}.
3. Tipo de evaluación: {tipo}.
4. Utiliza únicamente la información del documento.
5. No inventes datos ni conceptos.
6. Evita repetir preguntas.
7. Las preguntas deben ser claras y comprensibles.
8. Las respuestas deben ser correctas y estar respaldadas
   por el contenido del documento.
9. Responde exclusivamente con JSON válido.
10. No incluyas bloques Markdown ni explicaciones fuera
    del JSON.

FORMATO DE LAS PREGUNTAS:

{formato}

ESTRUCTURA JSON DE CADA PREGUNTA:

{estructura}

La respuesta final debe tener esta estructura:

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
            "educativas. Trabaja exclusivamente con el contexto "
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

    # Limpiar posibles bloques Markdown.
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
            "La IA no devolvió preguntas válidas. Intenta nuevamente."
        )

    # Validar estructura básica de cada pregunta.
    for i, pregunta in enumerate(preguntas, start=1):

        if not isinstance(pregunta, dict):
            raise ValueError("Formato incorrecto en una pregunta.")

        if not pregunta.get("pregunta"):
            raise ValueError("Hay una pregunta sin contenido.")

        if not pregunta.get("respuesta"):
            raise ValueError("Hay una pregunta sin respuesta.")

        opciones = pregunta.get("opciones", {})

        if not isinstance(opciones, dict):
            raise ValueError("Formato incorrecto de las opciones.")

        if tipo == "Opción múltiple":
            if not all(k in opciones for k in ["A", "B", "C", "D"]):
                raise ValueError("Una pregunta no tiene cuatro opciones.")

            if pregunta["respuesta"] not in ["A", "B", "C", "D"]:
                raise ValueError("Una respuesta de opción múltiple es inválida.")

        elif tipo == "Verdadero o falso":
            if pregunta["respuesta"] not in ["A", "B"]:
                raise ValueError("Una respuesta de verdadero/falso es inválida.")

        pregunta["numero"] = i

    if len(preguntas) != numero_preguntas:
        raise ValueError(
            f"Se solicitaron {numero_preguntas} preguntas, "
            f"pero la IA generó {len(preguntas)}. "
            "Intenta generar nuevamente."
        )

    return {
        "titulo": resultado.get(
            "titulo",
            "Evaluación de conocimientos"
        ),
        "preguntas": preguntas
    }


# ==========================================================
# CREAR DOCUMENTO PDF
# ==========================================================

def crear_pdf(evaluacion, nombre_documento, incluir_respuestas=False):

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
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=10
    )

    estilo_subtitulo = ParagraphStyle(
        "Subtitulo",
        parent=estilos["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20
    )

    estilo_pregunta = ParagraphStyle(
        "Pregunta",
        parent=estilos["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#172554"),
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
            f"Material de referencia: {html.escape(nombre_documento)}",
            estilo_subtitulo
        )
    )

    elementos.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#cbd5e1")
        )
    )

    elementos.append(Spacer(1, 20))

    if not incluir_respuestas:

        elementos.append(
            Paragraph("Nombre del estudiante: ______________________________", estilo_texto)
        )

        elementos.append(
            Paragraph("Fecha: _____________________     Curso: _____________________", estilo_texto)
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
                        f"{html.escape(str(letra))}. "
                        f"{html.escape(str(opcion))}",
                        estilo_texto
                    )
                )

        elif not incluir_respuestas:

            elementos.append(
                Paragraph(
                    "Respuesta: __________________________________________",
                    estilo_texto
                )
            )

            elementos.append(
                Paragraph(
                    "_____________________________________________________",
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
                    f"<b>Respuesta correcta / esperada:</b> {respuesta}",
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
# BOTÓN GENERAR EVALUACIÓN
# ==========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">✨ Genera tu evaluación</div>',
    unsafe_allow_html=True
)

st.write(
    "Cuando hayas introducido tu clave y seleccionado "
    "el documento, puedes comenzar."
)

if not api_key:

    st.warning(
        "🔑 Debes introducir tu clave API de OpenAI "
        "en el panel izquierdo para generar tu evaluación."
    )

elif pdf is None:

    st.info(
        "📄 Primero debes cargar un documento PDF."
    )

else:

    st.success(
        f"Documento preparado: {pdf.name}"
    )

    if st.button(
        "✨ GENERAR EVALUACIÓN",
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

            st.toast(
                f"Documento dividido en {fragmentos} fragmentos."
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

            st.success(
                "¡Tu evaluación se ha generado correctamente!"
            )

        except Exception as error:

            st.error(
                "No fue posible generar la evaluación."
            )

            st.exception(error)


# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

if st.session_state.evaluacion:

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

    for pregunta in evaluacion["preguntas"]:

        with st.container(border=True):

            st.markdown(
                f"**Pregunta {pregunta['numero']}**"
            )

            st.write(pregunta["pregunta"])

            opciones = pregunta.get("opciones", {})

            if opciones:

                for letra, opcion in opciones.items():

                    st.write(f"**{letra}.** {opcion}")

            else:

                st.text_area(
                    "Espacio para responder",
                    key=f"respuesta_{pregunta['numero']}",
                    disabled=True
                )

    st.markdown("---")

    st.markdown("### 📥 Descargar documentos")

    st.write(
        "Puedes descargar el examen para tus estudiantes "
        "y la clave de respuestas para uso del profesor."
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
            label="🔑 Descargar clave de respuestas",
            data=respuestas_pdf,
            file_name="Respuestas_EvaluaPro.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with st.expander("🔍 Consultar respuestas correctas"):

        for pregunta in evaluacion["preguntas"]:

            st.markdown(
                f"**Pregunta {pregunta['numero']}**"
            )

            st.write(
                f"Respuesta: {pregunta['respuesta']}"
            )

            st.write(
                f"Explicación: {pregunta.get('explicacion', '')}"
            )

            st.divider()

    if st.button(
        "🔄 Crear una nueva evaluación",
        use_container_width=True
    ):

        st.session_state.evaluacion = None
        st.session_state.nombre_documento = None
        st.session_state.configuracion = None

        st.rerun()


# ==========================================================
# PIE DE PÁGINA
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:15px; color:#64748b;">
        <b>EvaluaPro</b><br>
        Herramienta inteligente para la creación de evaluaciones.<br>
        <small>Desarrollado con Python, Streamlit y tecnología de IA.</small>
    </div>
    """,
    unsafe_allow_html=True
)
