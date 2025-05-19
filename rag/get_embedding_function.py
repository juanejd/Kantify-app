from langchain_aws import BedrockEmbeddings


def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="bedrock-admin", region_name="us-east-1"
    )
    return embeddings
