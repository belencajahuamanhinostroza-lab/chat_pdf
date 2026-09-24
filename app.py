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


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="RAG - Pregúntale a tu PDF",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG — Pregúntale a tu PDF")
st.caption("Generación Aumentada por Recuperación")

st.write("Versión de Python:", platform.python_version())


# --------------------------------------------------
# ESTILOS
# --------------------------------------------------

st.markdown("""
<style>

.document-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.answer-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f7f9fc;
    border: 1px solid #d
