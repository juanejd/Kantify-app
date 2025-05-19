from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# ruta de la carpeta data
DATA_PATH = "data"


def load_documents():
    """
    Cargar los documentos de la carpeta data
    retorna un diccionario con el contenido de texto en cada pagina del PDF
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """
    Divide el texto en fragmentos más pequeños y manejables. Cada fragmento tiene 1000 letras y comparte 200 letras con el fragmento anterior para no perder el hilo del texto
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_chunks() -> list[Document]:
    """
    Funcion para obtener los chunks de los documentos
    """
    documents = load_documents()
    chunks = split_documents(documents)
    return chunks
