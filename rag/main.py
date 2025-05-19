import os
from dotenv import load_dotenv

from charge_data import get_chunks
from chroma_db import add_to_chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Cargar variables de entorno desde .env
load_dotenv()

# Verificar si existe la API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "No se encontró la API key de OpenAI. Por favor, asegúrate de tener un archivo .env con OPENAI_API_KEY"
    )

# Obtenemos los chunks de los documentos
processed_documents = get_chunks()

# Obtenemos el embedding model que transforma los documentos en vectores numericos
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# Agregamos los chunks a la base de datos vectorial
db = add_to_chroma(processed_documents, embedding_model)

# Creamos una query para probar el RAG
query = "A que se refieren con el 'primer deber' de la etica?"

# Buscamos los chunks mas relevantes para la query
docs = db.similarity_search_with_score(query, k=3)

# Unimos los chunks en un solo string
context = "\n\n---\n\n".join([doc.page_content for doc, _score in docs])

# Creamos el prompt para el modelo
PROMPT_TEMPLATE = """
You have to answer the following question based on the given context:
{context}
Answer the following question:{question}
Provide a detailed answer in Spanish.
Don't include non-relevant information.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context, question=query)

# Creamos el modelo
model = ChatOpenAI()
response = model.predict(prompt)
print(response)
