
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
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)


# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="EvaluaPro | Evaluaciones inteligentes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. ESTILOS VISUALES
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --verde-oscuro: #075B55;
    --verde-profundo: #034C47;
    --verde-medio: #0B796C;
    --verde-lima: #DDF18B;
    --fondo: #F7F9F2;
    --texto: #173F3B;
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

/* --------------------------------------------------
   BARRA LATERAL
   -------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #034C47 0%,
        #075B55 65%,
        #08766A 100%
    );
}

section[data-testid="stSidebar"] * {
    font-family: 'Manrope', sans-serif;
}

/* Textos de la barra lateral en blanco */

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption {
    color: #FFFFFF !important;
}

/* Títulos en verde lima */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #DDF18B !important;
    font-weight: 800;
}

/* --------------------------------------------------
   BOTÓN PARA CONTRAER Y DESPLEGAR LA BARRA
   -------------------------------------------------- */

/* Mantener visible el botón de la barra lateral */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarExpandButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: #DDF18B !important;
    color: #034C47 !important;
    border: 2px solid #DDF18B !important;
    border-radius: 10px !important;
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}

/* Icono del botón visible */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="stSidebarExpandButton"] svg {
    color: #034C47 !important;
    fill: none !important;
    stroke: #034C47 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Ocultar texto del botón, conservar icono */

section[data-testid="stSidebar"]
button[data-testid="stSidebarCollapseButton"] span,
button[data-testid="stSidebarExpandButton"] span {
    display: none !important;
}

/* --------------------------------------------------
   BARRA DE DESPLAZAMIENTO LATERAL
   -------------------------------------------------- */

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    scrollbar-color: #DDF18B #075B55 !important;
}

/* Chrome, Edge y Safari */

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {
    width: 10px !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
    background: #034C47 !important;
    border-radius: 10px;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
    background: #DDF18B !important;
    border-radius: 10px;
    border: 2px solid #034C47;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
    background: #FFFFFF !important;
}

/* --------------------------------------------------
   ÁREA PRINCIPAL
   -------------------------------------------------- */

/* Textos principales oscuros */

.stApp [data-testid="stMain"] label,
.stApp [data-testid="stMain"] p,
.stApp [data-testid="stMain"] .stMarkdown {
    color: #173F3B;
}

/* Títulos principales */

.section-title {
    color: #075B55 !important;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin-top: 22px;
    margin-bottom: 12px;
}

/* --------------------------------------------------
   CAJAS PARA ESCRIBIR
   -------------------------------------------------- */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    color: #222222 !important;
    background-color: #FFFFFF !important;
    -webkit-text-fill-color: #222222 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

/* Texto de ejemplo dentro de las cajas */

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #777777 !important;
    -webkit-text-fill-color: #777777 !important;
}

/* Campo de clave API */

section[data-testid="stSidebar"] .stTextInput input {
    color: #222222 !important;
    background-color: #FFFFFF !important;
    -webkit-text-fill-color: #222222 !important;
}

/* --------------------------------------------------
   QUITAR OJO DE LA CLAVE API
   -------------------------------------------------- */

/* Oculta únicamente el botón de mostrar/ocultar contraseña */

section[data-testid="stSidebar"]
[data-testid="stTextInput"] button {
    display: none !important;
}

/* --------------------------------------------------
   SELECTORES
   -------------------------------------------------- */

.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #222222 !important;
}

/* --------------------------------------------------
   SLIDERS
   -------------------------------------------------- */

section[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] [data-testid="stSlider"] div {
    color: #FFFFFF;
}

/* --------------------------------------------------
   BOTONES
   -------------------------------------------------- */

div.stButton > button {
    background: #075B55;
    color: #FFFFFF !important;
    border: 1px solid #075B55;
    border-radius: 30px;
    padding: 12px 22px;
    min-height: 47px;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    background: #0C756A;
    color: #DDF18B !important;
    border-color: #0C756A;
}

/* --------------------------------------------------
   BOTONES DE DESCARGA
   -------------------------------------------------- */

div.stDownloadButton > button {
    background: #DDF18B;
    color: #075B55 !important;
    border: 1px solid #DDF18B;
    border-radius: 30px;
    min-height: 47px;
    font-weight: 800;
    font-family: 'Manrope', sans-serif;
}

div.stDownloadButton > button:hover {
    background: #CBE879;
    color: #075B55 !important;
    border-color: #CBE879;
}

/* --------------------------------------------------
   CARGADOR PDF
   -------------------------------------------------- */

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #A6C68D;
    border-radius: 20px;
    padding: 20px;
}

[data-testid="stFileUploader"] * {
    color: #173F3B;
}

/* --------------------------------------------------
   MENSAJES Y MÉTRICAS
   -------------------------------------------------- */

.stAlert {
    border-radius: 14px;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E5EBDD;
}

hr {
    border-color: #DFE9D7;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. ENCABEZADO PRINCIPAL
# ==========================================================

st.html("""
<div style="
    background: linear-gradient(
        125deg,
        #034C47 0%,
        #08736A 55%,
        #15947A 100%
    );
    padding: 42px;
    border-radius: 25px;
    margin-bottom: 32px;
    box-shadow: 0 12px 30px rgba(3,76,71,0.12);
    font-family: 'Manrope', sans-serif;
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
        margin: 0 0 12px 0;
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
    ">
        ✦ APRENDER · EVALUAR · MEJORAR
    </div>

</div>
""")


# ==========================================================
# 4. BARRA LATERAL
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
        help="Introduce una clave API válida de OpenAI."
    ).strip()

    if api_key:

        st.success("✓ Clave API introducida")

        st.html("""
        <div style="
            background:#E8F5DF;
            border:1px solid #B9D99D;
            border-radius:12px;
            padding:13px;
            margin-top:8px;
            color:#075B55;
            font-size:13px;
            font-weight:700;
        ">
            ✓ Ya puedes cargar tu documento PDF
            y generar tu evaluación.
        </div>
        """)

    else:

        st.warning("🔑 Introduce tu clave API para comenzar.")

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

    st.html("""
    <div style="
        text-align:center;
        color:#DDF18B;
        font-size:11px;
        font-weight:700;
        padding:10px;
    ">
        EVALUAPRO · EDUCACIÓN INTELIGENTE
    </div>
    """)


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
# 6. CARGA DEL PDF
# ==========================================================

st.markdown(
    '<div class="section-title">📄 Carga tu documento PDF</div>',
    unsafe_allow_html=True
)

st.write(
    "Selecciona tu material de clase para crear "
    "una evaluación inteligente."
)

if api_key:
    st.success(
        "✓ Conexión preparada. Ya puedes seleccionar tu PDF."
    )
else:
    st.warning(
        "🔑 Primero introduce tu clave API de OpenAI "
        "en el menú lateral."
    )

pdf = st.file_uploader(
    "Selecciona tu material de clase",
    type=["pdf"],
    help="Sube un documento PDF con texto seleccionable."
)

if pdf is not None:

    st.success(f"✓ Documento cargado: {pdf.name}")

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
# 7. EXTRAER TEXTO DEL PDF
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
# 8. CREAR BASE DE CONOCIMIENTO
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
# 9. GENERAR EVALUACIÓN
# ==========================================================

def generar_evaluacion(
    base,
    api_key,
    numero_preguntas,
    dificultad,
    tipo
):

    docs = base.similarity_search(
        "Conceptos principales, definiciones, características, "
        "explicaciones, ejemplos y datos importantes del documento.",
        k=min(15, base.index.ntotal)
    )

    contexto = "\n\n---\n\n".join(
        doc.page_content for doc in docs
    )

    if tipo == "Opción múltiple":

        formato = """
Cada pregunta debe tener cuatro opciones: A, B, C y D.
Debe existir una sola respuesta correcta.
La respuesta debe ser A, B, C o D.
"""

        ejemplo_opciones = """
"opciones": {
    "A": "Primera opción",
    "B": "Segunda opción",
    "C": "Tercera opción",
    "D": "Cuarta opción"
},
"respuesta": "A"
"""

    elif tipo == "Verdadero o falso":

        formato = """
Cada pregunta debe ser una afirmación que pueda
clasificarse como Verdadero o Falso.
La respuesta debe ser A para Verdadero o B para Falso.
"""

        ejemplo_opciones = """
"opciones": {
    "A": "Verdadero",
    "B": "Falso"
},
"respuesta": "A"
"""

    else:

        formato = """
Cada pregunta debe ser abierta y permitir que el estudiante
responda con sus propias palabras.
Incluye una respuesta esperada.
"""

        ejemplo_opciones = """
"opciones": {},
"respuesta": "Respuesta esperada"
"""

    instrucciones = f"""
Eres un experto en pedagogía y diseño de evaluaciones.

Crea un examen educativo a partir del documento.

REGLAS:

1. Genera exactamente {numero_preguntas} preguntas.
2. Dificultad: {dificultad}.
3. Modalidad: {tipo}.
4. Utiliza únicamente información del documento.
5. No inventes datos ni conceptos.
6. Evita repetir preguntas.
7. Las preguntas deben ser claras.
8. Las respuestas deben estar respaldadas por el documento.
9. Devuelve exclusivamente JSON válido.
10. No utilices bloques Markdown.

REGLAS ESPECÍFICAS:

{formato}

FORMATO DE CADA PREGUNTA:

{{
    "numero": 1,
    "pregunta": "Texto de la pregunta",
    {ejemplo_opciones},
    "explicacion": "Explicación de la respuesta"
}}

ESTRUCTURA FINAL:

{{
    "titulo": "Evaluación de conocimientos",
    "preguntas": []
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
            "Intenta nuevamente."
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
            raise ValueError("Formato incorrecto de opciones.")

        if tipo == "Opción múltiple":

            if not all(
                letra in opciones
                for letra in ["A", "B", "C", "D"]
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
# 10. CREAR PDF
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
            "Material de referencia: "
            + html.escape(nombre_documento),
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
# 11. CONTROL DEL DOCUMENTO
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
# 12. GENERAR EVALUACIÓN
# ==========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">✨ Genera tu evaluación</div>',
    unsafe_allow_html=True
)

if not api_key:

    st.warning(
        "🔑 Debes introducir tu clave API de OpenAI "
        "en el menú lateral para continuar."
    )

elif pdf is None:

    st.info(
        "📄 Carga tu documento PDF para poder generar "
        "tu evaluación."
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
                "No fue posible generar la evaluación. "
                "Comprueba tu clave API y vuelve a intentarlo."
            )

            st.exception(error)


# ==========================================================
# 13. MOSTRAR EVALUACIÓN
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
                    st.write(f"**{letra}.** {opcion}")

            else:

                st.markdown("_Respuesta abierta_")

                st.text_area(
                    "Espacio para responder",
                    key=f"respuesta_{pregunta['numero']}",
                    disabled=True
                )


    # ======================================================
    # 14. DESCARGAR DOCUMENTOS
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
    # 15. CONSULTAR RESPUESTAS
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
    # 16. NUEVA EVALUACIÓN
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
# 17. PIE DE PÁGINA
# ==========================================================

st.markdown("---")

st.html("""
<div style="
    text-align: center;
    padding: 30px 10px;
    color: #71817B;
    font-family: 'Manrope', sans-serif;
">

    <div style="
        font-size: 22px;
        font-weight: 800;
        color: #075B55;
        margin-bottom: 8px;
    ">
        EvaluaPro.
    </div>

    <div style="
        font-size: 13px;
        font-weight: 600;
        color: #075B55;
    ">
        Aprende · Evalúa · Mejora
    </div>

    <div style="
        font-size: 11px;
        margin-top: 12px;
        color: #71817B;
    ">
        Plataforma inteligente para la creación
        de evaluaciones educativas.
    </div>

    <div style="
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #075B55, #DDF18B);
        border-radius: 10px;
        margin: 18px auto 0 auto;
    "></div>

</div>
""")
