%%writefile app.py
import streamlit as st
import pdfplumber
import pandas as pd
import time
from keybert import KeyBERT

st.set_page_config(page_title="Professional Keyword Extractor", layout="centered")

@st.cache_resource
def load_nlp_model():
    return KeyBERT(model='all-MiniLM-L6-v2')

kw_model = load_nlp_model()

def extract_data_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

st.title("Comprehensive Keyword Extractor")
uploaded_file = st.file_uploader("Choose PDF", type=['pdf'])

if uploaded_file:
    with st.status("Processing...", expanded=True):
        document_text = extract_data_from_pdf(uploaded_file)
        if document_text.strip():
            keywords = kw_model.extract_keywords(
                document_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                use_mmr=True,
                diversity=0.7,
                top_n=20
            )
            st.subheader("Extracted Results")
            df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
            st.table(df)
