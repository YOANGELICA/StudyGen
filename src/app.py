import os
import dotenv #type: ignore
import streamlit as st #type: ignore
import re

from rag import (
    load_pdf,
    load_web_url,
    load_youtube,
    embed_documents,
    extract_main_topics,
    generate_questions
)

dotenv.load_dotenv()

st.set_page_config(page_title="StudyGen", page_icon="📘")
st.title("📘 StudyGen – Guía de Estudios Personalizada")

st.markdown("""
Sube tus materiales de clase o pega enlaces. Puedes combinar PDFs, páginas web o videos de YouTube (máx. 5 min).
Selecciona un tema para generar preguntas tipo examen.
""")

# === Carga de contenido ===
st.header("📚 Agrega contenido")

with st.expander("📄 Subir archivo PDF"):
    uploaded_file = st.file_uploader("Selecciona un archivo PDF", type=["pdf"])
    if uploaded_file is not None:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Procesando PDF..."):
            splits = load_pdf("temp_uploaded.pdf")
            st.session_state.vector_store = embed_documents(splits, st.session_state.get("vector_store"))
            st.session_state.document_text = st.session_state.get("document_text", "") + "\n".join([s.page_content for s in splits])
        os.remove("temp_uploaded.pdf")
        st.success("PDF cargado correctamente.")

with st.expander("🌐 Pegar enlace de página web"):
    url = st.text_input("Enlace web")
    if st.button("Agregar página web"):
        if url:
            with st.spinner("Procesando página web..."):
                docs = load_web_url(url)
                st.session_state.vector_store = embed_documents(docs, st.session_state.get("vector_store"))
                st.session_state.document_text = st.session_state.get("document_text", "") + "\n".join([d.page_content for d in docs])
            st.success("Página agregada correctamente.")

with st.expander("▶️ Pegar enlace de YouTube (máx. 5 min)"):
    yt_url = st.text_input("Enlace de video")
    if st.button("Agregar video"):
        if yt_url:
            with st.spinner("Procesando video..."):
                try:
                    docs = load_youtube(yt_url)
                    if not docs:
                        raise ValueError("No se pudo obtener la transcripción del video.")
                    st.session_state.vector_store = embed_documents(docs, st.session_state.get("vector_store"))
                    st.session_state.document_text = st.session_state.get("document_text", "") + "\n".join([d.page_content for d in docs])
                    st.success("Video agregado correctamente.")
                except Exception as e:
                    st.error("❌ No se pudo procesar el video. Asegúrate de que:")
                    st.markdown("""
                    - El video tiene subtítulos (automáticos o manuales).
                    - La duración es menor a 5 minutos.
                    - El video no es privado o restringido por edad/región.
                    - La URL es válida (ej: `https://www.youtube.com/watch?v=abc123`).
                    """)
                    st.exception(e)

# === Modo avanzado ===
st.sidebar.header("⚙️ Opciones avanzadas (docentes)")
modo_avanzado = st.sidebar.checkbox("Mostrar configuración del modelo")

if modo_avanzado:
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)
else:
    temperature, top_p, top_k = 0.7, 0.9, 40

# === Extraer temas principales ===
if "vector_store" in st.session_state and "temas" not in st.session_state:
    with st.spinner("🧠 Extrayendo temas principales..."):
        st.session_state.temas = extract_main_topics(st.session_state.document_text)

# === Mostrar temas y generar preguntas ===
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

# === Mostrar preguntas con respuestas ocultas ===
if "resultado" in st.session_state:
    st.subheader("📋 Preguntas generadas")

    blocks = re.split(r"\n(?=\d+\.\s*Pregunta:)", st.session_state.resultado)
    
    for i, block in enumerate(blocks, start=1):
        if not block.strip():
            continue

        pregunta_match = re.search(r"Pregunta:\s*(.*)", block)
        respuesta_match = re.search(r"Respuesta:\s*(.*)", block)
        explicacion_match = re.search(r"Explicación:\s*(.*)", block)

        pregunta = pregunta_match.group(1).strip() if pregunta_match else f"Pregunta {i}"
        respuesta = respuesta_match.group(1).strip() if respuesta_match else "No encontrada"
        explicacion = explicacion_match.group(1).strip() if explicacion_match else "No disponible"

        st.markdown(f"**❓ Pregunta {i}:** {pregunta}")
        with st.expander("Mostrar respuesta y explicación"):
            st.markdown(f"**✅ Respuesta:** {respuesta}")
            st.markdown(f"**💡 Explicación:** {explicacion}")