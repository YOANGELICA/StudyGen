import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    YoutubeLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return _split_docs(docs)


def load_web_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return _split_docs(docs)


def load_youtube(url: str):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    return _split_docs(docs)
# Video que funciona: https://www.youtube.com/watch?v=5MgBikgcWnY


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_documents(docs)


def embed_documents(splits, existing_store=None, persist_dir: str = "../chroma_langchain_db", collection_name: str = "collection"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if existing_store:
        existing_store.add_documents(splits)
        return existing_store
    else:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        vector_store.add_documents(splits)
        return vector_store


def extract_main_topics(docs: str, llm=None):
    if llm is None:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.3)

    prompt = f"""
        Analiza las siguientes fuentes y extrae exactamente **3 temas o conceptos principales** del contenido.

        Devuélvelos **ESPECIFICAMENTE** en forma de lista numerada, SIN introducciones o conclusiones.

        Texto:
        {docs[:3000]}
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    temas = response.content.strip().split("\n")
    temas = [t.strip("-•1234567890. ").strip() for t in temas if t.strip()]
    return temas[:3]


def generate_questions(selected_topic: str, vector_store, llm=None, top_p=0.9, temperature=0.7, top_k=40):
    if llm is None:
        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

    results = vector_store.similarity_search_with_score(selected_topic, k=4)
    retrieved_docs = "\n\n".join([doc.page_content for doc, score in results])

    prompt = f"""
        Genera exactamente 3 preguntas tipo examen sobre el tema: "{selected_topic}" usando únicamente el siguiente contenido académico.

        Sigue esta estructura EXACTA, sin encabezados, sin indicaciones adicionales ni numeraciones duplicadas. 
        Cada pregunta debe seguir el siguiente patrón (NO LO MODIFIQUES):

        1. Pregunta: <enunciado (pregunta de análisis del tema)>
        Tipo: Ensayo 
        Respuesta: <respuesta>
        Explicación: <explicación>

        2. Pregunta: <enunciado>
        Tipo: Selección múltiple 
        Alternativas:
        A) ...
        B) ...
        C) ...
        D) ...
        Respuesta: <respuesta>
        Explicación: <explicación>
        
        3. Pregunta: <enunciado>
        Tipo: Verdadero/Falso
        Respuesta: <Verdadero o Falso>
        Explicación: <explicación>

        Las **preguntas tipo ensayo** (o "Tipo: Ensayo") requieren respuestas desarrolladas, argumentativas o explicativas, con múltiples oraciones.
        Las **preguntas tipo selección múltiple** (o "Tipo: Selección múltiple") deben incluir 4 alternativas, de las cuales solo una es correcta.
        Las **preguntas tipo verdadero/falso** (o "Tipo: Verdadero/Falso") deben ser claras y poder ser respondidas con "Verdadero" o "Falso".

        Ahora genera tus las preguntas a partir del siguiente contenido:
        {retrieved_docs}
        """

    messages = [HumanMessage(content=prompt)]
    stream = llm.stream(messages)

    metadata = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }

    return stream, metadata