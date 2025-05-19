import os
import shutil
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document

CHROMA_PATH = "chroma"


def add_to_chroma(chunks: list[Document], embedding_model) -> Chroma:
    """
    Funcion para agregar los chunks a la base de datos vectorial
    """
    # En caso de que exista la carpeta, se elimina
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            print(f"Error al eliminar la carpeta {CHROMA_PATH}: {e}")

    # Inicializar un objeto que contendra la base de datos vectorial
    db = Chroma.from_documents(
        chunks,
        persist_directory=CHROMA_PATH,
        embedding=embedding_model,
    )
    print(f"Base de datos vectorial creada en {CHROMA_PATH}")
    return db
