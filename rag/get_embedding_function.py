from langchain_openai import OpenAIEmbeddings


def get_embedding_function():
    """
    Funcion para convertir los datos de los documentos en vectores
    y guardar la informacion en una base de datos vectorial
    """
    # Embedding de OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    return embeddings
