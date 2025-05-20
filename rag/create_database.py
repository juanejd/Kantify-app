from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil

# Cargar variables de entorno desde .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# ruta de la carpeta data
CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    get_chunks()


def get_chunks() -> list[Document]:
    """
    Funcion para obtener los chunks de los documentos
    """
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)


def load_documents():
    """
    Cargar los documentos de la carpeta data
    retorna un diccionario con el contenido de texto en cada pagina del PDF
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    return documents


def split_documents(documents: list[Document]):
    """
    Divide el texto en fragmentos más pequeños y manejables. Cada fragmento tiene 800 letras y comparte 100 letras con el fragmento anterior para no perder el hilo del texto
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Funcion para guardar los chunks en la base de datos vectorial
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # crear una nueva base de datos de los documentos
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()
