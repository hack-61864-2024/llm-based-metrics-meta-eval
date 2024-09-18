import os
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

load_dotenv()

#### - CONFIGURE  AZURE OPENAI - ####
def configure_azure():
    """
    Configures Azure OpenAI services for use with LangChain.

    This function loads Azure OpenAI configuration settings from environment variables, initializes 
    the `AzureChatOpenAI` model and `AzureOpenAIEmbeddings` using these settings, and returns them.

    Environment Variables:
        AZURE_OPENAI_ENDPOINT (str): The base URL for the Azure OpenAI endpoint.
        AZURE_OPENAI_MODEL_DEPLOYMENT (str): The name of the model deployment for Azure OpenAI.
        AZURE_OPENAI_MODEL_NAME (str): The name of the Azure OpenAI model.
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT (str): The name of the embedding deployment for Azure OpenAI.
        AZURE_OPENAI_EMBEDDING_NAME (str): The name of the embedding model for Azure OpenAI.

    Returns:
        tuple: A tuple containing:
            - azure_model (AzureChatOpenAI): Configured AzureChatOpenAI model for chat operations.
            - azure_embeddings (AzureOpenAIEmbeddings): Configured AzureOpenAIEmbeddings for embedding operations.

    """

    
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