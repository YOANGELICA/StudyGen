import os
import dotenv
import streamlit as st

from rag import load_pdf, embed_documents, extract_main_topics, generate_questions

dotenv.load_dotenv()

# Configuración
st.set_page_config(page_title="StudyGen", page_icon="📘")
st.title("📘 StudyGen – Guía de Estudios Personalizada")

st.markdown("""
Sube tus materiales de clase (PDF). Elige uno de los 3 temas principales y podrás generar preguntas tipo examen sobre el tema que elijas.
""")

st.sidebar.header("📄 Subir archivo PDF")
uploaded_file = st.sidebar.file_uploader("Archivo PDF", type=["pdf"])


# TODO -> NOTA: Tal vez sea mejor quitar la opcion de los parametros, ya que el usuario final no sabe que es lo que hace cada uno ?
st.sidebar.header("🧠 Parámetros del modelo")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)

if uploaded_file is not None:
    if "vector_store" not in st.session_state:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("📄 Procesando documento..."):
            splits = load_pdf("temp_uploaded.pdf")
            st.session_state.vector_store = embed_documents(splits)
            st.session_state.document_text = "\n".join([s.page_content for s in splits])

        os.remove("temp_uploaded.pdf")

# Deteccion de temas principales
if "vector_store" in st.session_state and "temas" not in st.session_state:
    with st.spinner("🧠 Extrayendo temas principales..."):
        # guardar los temas generados en el sesion state
        st.session_state.temas = extract_main_topics(st.session_state.document_text)

# Mostrar temas
if "temas" in st.session_state:
    st.subheader("📌 Temas principales detectados")
    selected_topic = st.radio("Selecciona un tema para generar preguntas:", st.session_state.temas)

    if st.button("🎯 Generar preguntas sobre el tema"):
        with st.spinner("✍️ Generando preguntas..."):
            stream, metadata = generate_questions(
                selected_topic,
                st.session_state.vector_store,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            st.session_state.resultado = "".join([chunk.content for chunk in stream])

# Mostrar preguntas
if "resultado" in st.session_state:
    st.subheader("📋 Preguntas generadas")
    st.markdown(st.session_state.resultado)
