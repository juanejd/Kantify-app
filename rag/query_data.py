import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

CHROMA_PATH = "chroma"

# Plantilla del prompt dandole el contexto y el prompt escrito
PROMPT_TEMPLATE = """
You have to answer the following question based on the given context:
{context}
Answer the following question:{question}
Provide a detailed answer in Spanish.
Don't include non-relevant information.
"""


def main():
    # Se crea un CLI para escribir la consulta en la terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Preparamos la base de datos
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH,
    )

    # Buscamos en la base de datos los documentos mas relevantes en base a la consulta y crear el contexto
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return

    # Preparamos el prompt para el modelo de OpenAI
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Obtenemos la respuesta del modelo de OpenAI
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # Obtenemos las fuentes de los documentos
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
