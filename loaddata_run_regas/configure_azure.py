import os
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

load_dotenv()

#### - CONFIGURE  AZURE OPENAI - ####
def configure_azure():
    azure_configs = {
    "base_url": os.getenv('AZURE_OPENAI_ENDPOINT'),
    "model_deployment": os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT'),
    "model_name": os.getenv('AZURE_OPENAI_MODEL_NAME'),
    "embedding_deployment": os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
    "embedding_name": os.getenv('AZURE_OPENAI_EMBEDDING_NAME'),
    }

    azure_model = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    )

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
    azure_embeddings = AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    )
    
    return azure_model,azure_embeddings