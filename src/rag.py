import os
import time
# import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage


def load_pdf(file_relative_path: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(base_dir, file_relative_path))

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Docs length: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, add_start_index=True
    )
    all_splits = splitter.split_documents(docs)
    print(f"all_splits length: {len(all_splits)}")

    return all_splits

def embed_documents(splits, persist_dir: str = "../chroma_langchain_db", collection_name: str = "collection"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_path = os.path.abspath(os.path.join(base_dir, persist_dir))

    # if os.path.exists(persist_path):
    #     shutil.rmtree(persist_path)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_path
    )

    # if os.path.exists(persist_path) and len(vector_store.get()["documents"]) > 0:
    #     print("Vector store ya existente con documentos.")
    #     return vector_store
    
    vector_store.add_documents(splits)

    print(f"Indexed {len(splits)} chunks.")

    return vector_store


def retrieval(vector_store, query: str):
    results = vector_store.similarity_search_with_score(query)
    retrieved_docs = "\n\n".join([doc.page_content for doc, score in results])
    
    return retrieved_docs, len(results)

def generate_response(retrieved_docs: str, query: str, top_p=0.9, temperature=0.7, top_k=40):
    
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    messages = [
        HumanMessage(
            content=f"Respond the question based on the documents:\n\n{retrieved_docs}\n\n"
                    f"Question: {query}"
        )
    ]

    stream = llm.stream(messages)

    metadata = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
    return stream, metadata



# extraer 3 temas principales
def extract_main_topics(docs: str, llm=None):
    if llm is None:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.3)

    prompt = f"""
        Analiza el siguiente texto académico y extrae exactamente **3 temas o conceptos principales** del contenido.

        Devuélvelos en forma de lista numerada, SIN agregar introducciones antes de mencionar los conceptos.

        Texto:
        {docs[:3000]}
        """
    # TODO: revisar si es necesario el truncado a 3000 caracteres, buscar una mejor manera de hacerlo porque puede sesgar la respuesta

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    temas = response.content.strip().split("\n")
    temas = [t.strip("-•1234567890. ").strip() for t in temas if t.strip()] # quitar numeracion, guiones y esapcios en blanco

    return temas[:3]


def generate_questions(selected_topic: str, vector_store, llm=None, top_p=0.9, temperature=0.7, top_k=40):
    if llm is None:
        llm = ChatOllama(
            model="llama3.2:latest",
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

    # Retrieval usando el tema elegido para el query
    results = vector_store.similarity_search_with_score(selected_topic, k=4)
    retrieved_docs = "\n\n".join([doc.page_content for doc, score in results])

    prompt = f"""
        Genera 3 preguntas tipo examen sobre el tema: "{selected_topic}", usando únicamente el siguiente contenido académico. 
        Incluye una respuesta correcta y una breve explicación para cada una. NO incluyas introducciones antes de las preguntas.
        - 1 pregunta tipo ensayo.
        - 1 pregunta de selección múltiple.
        - 1 pregunta de verdadero/falso.

        Contenido:
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
