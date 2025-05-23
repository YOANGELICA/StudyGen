# StudyGen: AI-Powered Study Question Generator

StudyGen es una aplicación de estudio interactiva basada en RAG (Retrieval-Augmented Generation) que permite a los usuarios generar preguntas tipo examen a partir de contenido propio como PDFs, páginas web y videos de YouTube. Usa LangChain para la orquestación, Chroma como base de datos vectorial, y modelos alojados localmente a través de Ollama.

---

## 🧠 ¿Por qué llama3.2 + nomic-embed-text?

El modelo `llama3.2:latest` fue elegido por su buena calidad en generación de texto educativo y compatibilidad directa con Ollama. Para embeddings, se utiliza `nomic-embed-text` por su equilibrio entre rendimiento semántico y facilidad de uso local.

---

## 🚀 Cómo clonar y ejecutar el proyecto

### 1. Clona el repositorio
```bash
git clone https://github.com/YOANGELICA/StudyGen
cd StudyGen
````

### 2. Instala y ejecuta Ollama

Si no tienes Ollama instalado, primero instálalo:
👉 [https://ollama.com/download](https://ollama.com/download)

Luego abre Ollama para que quede corriendo en segundo plano.


### 3. Descarga los modelos necesarios

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

* `llama3` se utiliza para generar preguntas.
* `nomic-embed-text` para generar los embeddings vectoriales.


### 4. Instala dependencias (usando `uv` o `pip`)

Este proyecto usa `uv`.

```bash
pip install uv
uv pip install -r pyproject.toml
```

### 5. Crea el archivo `.env`

En la raíz del proyecto, crea un archivo `.env` con el siguiente contenido:

```env
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
USER_AGENT=studygen
```

Puedes obtener una API key gratuita en [https://smith.langchain.com](https://smith.langchain.com)


### 6. Ejecuta la aplicación

```bash
uv run streamlit run src/app.py
```

Una vez en marcha, podrás:

* Subir contenido (PDF, web, YouTube)
* Extraer automáticamente temas principales
* Generar preguntas de ensayo, selección múltiple y verdadero/falso
* Visualizar respuestas explicadas en una interfaz intuitiva

---

## 📁 Estructura del proyecto

```
StudyGen/
├── src/
│   ├── app.py                # Interfaz Streamlit
│   └── rag.py                # Lógica de carga, RAG y embeddings
├── .gitignore
├── .python-version
├── README.md
└── pyproject.toml            # Dependencias
```

---

## 🛠 Tecnologías usadas

* **LangChain** – Orquestación de RAG
* **Chroma** – Almacenamiento vectorial local
* **Ollama** – LLMs y embeddings en entorno local
* **Streamlit** – Interfaz web interactiva

