from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_function():
    """
    Funcion para obtener el embedding de HuggingFace
    """
    model_name = "hkunlp/instructor-xl"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return hf
