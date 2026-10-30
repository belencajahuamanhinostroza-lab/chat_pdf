
# ==========================================================
# DISEÑO VISUAL EVALUAPRO - ESTILO VERDE MODERNO
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --verde-oscuro: #075B55;
    --verde-profundo: #034C47;
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
    background: var(--fondo);
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
    position: relative;
    overflow: hidden;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: white;
    letter-spacing: -1.5px;
    margin-bottom: 12px;
}

.hero p {
    color: #E5F5DB;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
}

/* TÍTULOS */

.section-title {
    color: var(--verde-oscuro);
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.7px;
    margin-top: 25px;
    margin-bottom: 16px;
}

/* TARJETAS */

.info-card {
    background: white;
    border: 1px solid #E5EBDD;
    border-radius: 20px;
    padding: 23px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(7,91,85,0.04);
}

.info-card h3 {
    color: var(--verde-oscuro);
    font-size: 18px;
    font-weight: 800;
}

.info-card p {
    color: var(--gris);
    font-size: 13px;
}

/* TARJETAS DE PREGUNTAS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 18px;
    border-color: #E2EBDD;
}

/* BOTONES */

div.stButton > button {
    background: var(--verde-oscuro);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 12px 24px;
    min-height: 48px;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    transition: 0.25s;
}

div.stButton > button:hover {
    background: #0C756A;
    color: var(--verde-lima);
    border: none;
    transform: translateY(-2px);
}

/* DESCARGAS */

div.stDownloadButton > button {
    background: var(--verde-lima);
    color: var(--verde-oscuro);
    border: none;
    border-radius: 30px;
    min-height: 48px;
    font-weight: 800;
    font-family: 'Manrope', sans-serif;
}

div.stDownloadButton > button:hover {
    background: #CBE879;
    color: var(--verde-oscuro);
    border: none;
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
    color: var(--verde-lima);
    font-weight: 800;
}

/* CAMPOS DE TEXTO */

.stTextInput input,
.stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #DCE7D7;
}

/* SELECTORES */

.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px;
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

/* PIE DE PÁGINA */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)
