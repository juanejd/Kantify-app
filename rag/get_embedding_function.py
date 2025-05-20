from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

# Cargar las variables de entorno
load_dotenv()

# Obtener la API key de OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]


def main():
    # Obtener el embedding para una palabra
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")


if __name__ == "__main__":
    main()
