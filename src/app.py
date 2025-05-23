import os
import dotenv #type: ignore
import streamlit as st #type: ignore
import re
from datetime import datetime

from rag import (
    load_pdf,
    load_web_url,
    load_youtube,
    embed_documents,
    extract_main_topics,
    generate_questions
)

dotenv.load_dotenv()


def render_question(texto: str):
    blocks = re.split(r"\n(?=\d+\.\s*Pregunta:)", texto)

    for i, block in enumerate(blocks, start=1):
        if not block.strip():
            continue

        pregunta_match = re.search(r"Pregunta:\s*(.*)", block)
        alternativas_match = re.search(r"Alternativas:(.*?)(?:Respuesta:|Explicación:)", block, re.DOTALL)
        respuesta_match = re.search(r"Respuesta:\s*(.*)", block)
        explicacion_match = re.search(r"Explicación:\s*(.*)", block)

        pregunta = pregunta_match.group(1).strip() if pregunta_match else f"Pregunta {i}"
        alternativas = alternativas_match.group(1).strip() if alternativas_match else None
        respuesta = respuesta_match.group(1).strip() if respuesta_match else "No encontrada"
        explicacion = explicacion_match.group(1).strip() if explicacion_match else "No disponible"

        st.markdown(f"**❓ Pregunta {i}:** {pregunta}")
        if alternativas:
            st.markdown("**🔢 Alternativas:**")
            for linea in alternativas.splitlines():
                if linea.strip():
                    st.markdown(f"- {linea.strip()}")

        with st.expander("Mostrar respuesta y explicación"):
            st.markdown(f"**✅ Respuesta:** {respuesta}")
            st.markdown(f"**💡 Explicación:** {explicacion}")

if "document_blocks" not in st.session_state:
    st.session_state.document_blocks = []

st.set_page_config(page_title="StudyGen", page_icon="📘")
st.title("📘 StudyGen – Guía de Estudios Personalizada")

st.markdown("""
    Sube tus materiales de clase o pega enlaces. Puedes combinar PDFs, páginas web o videos de YouTube (máx. 5 min).
    Después, selecciona un tema para generar preguntas tipo examen.
    """)

# === Vista principal ===

st.session_state.vista = "principal"
st.session_state.seleccion_historial = None

# === Modo avanzado ===
st.sidebar.header("⚙️ Opciones avanzadas")
modo_avanzado = st.sidebar.checkbox("Mostrar configuración del modelo")

if modo_avanzado:
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)
else:
    temperature, top_p, top_k = 0.7, 0.9, 40
    
st.sidebar.header("🕓 Historial de sesiones")
if "historial" in st.session_state:
    for idx, item in enumerate(st.session_state.historial):
        # etiqueta = f"{item['tema']} ({item['timestamp'].split()[1]})"
        etiqueta = f"{item['timestamp']} – {item['tema']}"
        if st.sidebar.button(etiqueta[:30]+"...", key=f"ver_{idx}", ):
            st.session_state.vista = "historial"
            st.session_state.seleccion_historial = idx

if st.session_state.get("historial"):
    if st.sidebar.button("🗑️ Borrar historial"):
        st.session_state.confirm_delete = True

    if st.session_state.get("confirm_delete", False):
        with st.sidebar.warning("⚠️ ¿Estás seguro de que deseas borrar el historial?"):
            # st.warning("¿Estás seguro de que deseas borrar el historial?")
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Sí, borrar", key="yes_delete"):
                    st.session_state.historial = []
                    st.session_state.confirm_delete = False
                    st.sidebar.success("Historial borrado.")
                    st.rerun()
            if col2.button("Cancelar", key="cancel_delete"):
                    st.session_state.confirm_delete = False
                    


# === Vista de historial ===
if st.session_state.get("vista") == "historial":
    idx = st.session_state.seleccion_historial
    bloque = st.session_state.historial[idx]

    st.subheader(f"📋 Preguntas generadas sobre: {bloque['tema']}")
    blocks = re.split(r"\n(?=\d+\.\s*Pregunta:)", bloque["contenido"])

    render_question(bloque["contenido"])

    # Botón para volver a vista principal
    if st.button("🔙 Volver"):
        st.session_state.vista = "principal"
        st.session_state.seleccion_historial = None

else:
        
    with st.container(border=True):
        # === Carga de contenido ===
        st.header("📚 Agrega contenido")

        if "any_content_loaded" not in st.session_state:
            st.session_state.any_content_loaded = False

        # === PDF ===
        with st.expander("📄 Subir archivo PDF"):
            uploaded_file = st.file_uploader("Selecciona un archivo PDF", type=["pdf"], key="pdf_uploader")

            if uploaded_file is not None and "pdf_uploaded" not in st.session_state:
                st.session_state.pdf_uploaded = True 

                with open("temp_uploaded.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                with st.spinner("Procesando PDF..."):
                    splits = load_pdf("temp_uploaded.pdf")
                    st.session_state.vector_store = embed_documents(splits, st.session_state.get("vector_store"))
                    # st.session_state.document_text = st.session_state.get("document_text", "") + "\n".join([s.page_content for s in splits])
                    st.session_state.document_blocks.append("\n".join([s.page_content for s in splits]))

                os.remove("temp_uploaded.pdf")
                st.session_state.any_content_loaded = True
                st.success("PDF cargado correctamente.")

        # === Página Web ===
        with st.expander("🌐 Pegar enlace de página web"):
            url = st.text_input("Enlace web")

            if st.button("Agregar página web"):
                if url:
                    with st.spinner("Procesando página web..."):
                        docs = load_web_url(url)
                        st.session_state.vector_store = embed_documents(docs, st.session_state.get("vector_store"))
                        # st.session_state.document_text = st.session_state.get("document_text", "") + "\n".join([d.page_content for d in docs])
                        st.session_state.document_blocks.append("\n".join([d.page_content for d in docs]))
                    st.session_state.any_content_loaded = True
                    st.success("Página agregada correctamente.")

        # === YouTube ===
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
                            # st.session_state.document_text = st.session_state.get("document_text", "") + "\n".join([d.page_content for d in docs])
                            st.session_state.document_blocks.append("\n".join([d.page_content for d in docs]))
                            st.session_state.any_content_loaded = True
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
        
        left, middle, right = st.columns(3, vertical_alignment="bottom")

        with middle:
            iniciar = st.button("Iniciar", type="primary", use_container_width=True)

        if iniciar:
            if not st.session_state.get("any_content_loaded", False):
                st.warning("⚠️ Primero debes subir al menos una (1) fuente (PDF, video o página web).")
            else:
                left, middle, right = st.columns([1,2,1], vertical_alignment="center")
                with middle:
                    with st.spinner("🧠 Extrayendo temas principales..."):
                        st.session_state.temas = extract_main_topics(st.session_state.document_blocks)


    # === Mostrar temas y generar preguntas ===
    if "temas" in st.session_state:
        st.subheader("📌 Temas principales detectados")
        selected_topic = st.radio("Selecciona un tema para generar preguntas:", st.session_state.temas)

        left, middle, right = st.columns(3, vertical_alignment="bottom")

        with middle:
            if st.button("🎯 Generar preguntas sobre el tema", use_container_width=False):
                with st.spinner("✍️ Generando preguntas..."):
                    stream, metadata = generate_questions(
                        selected_topic,
                        st.session_state.vector_store,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )

                    resultado = "".join([chunk.content for chunk in stream])
                    st.session_state.resultado = resultado

                    # Guardar en historial
                    nuevo_bloque = {
                        "tema": selected_topic,
                        "contenido": resultado,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    if "historial" not in st.session_state:
                        st.session_state.historial = []
                    st.session_state.historial.insert(0, nuevo_bloque)
                    st.rerun()

    # === Mostrar preguntas con respuestas ocultas ===
    if "resultado" in st.session_state:
        st.subheader("📋 Preguntas generadas")
        
        render_question(st.session_state.resultado)
