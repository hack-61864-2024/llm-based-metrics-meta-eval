# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
from promptflow.core import tool
from ragas.metrics import faithfulness, answer_correctness
from datasets import Dataset
from ragas import evaluate
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from datasets import load_dataset

@tool
def evaluate_correctness(question: str, answer: str, ground_truth: str) -> float:
    """
    This tool processes the correctness of a single line.

    :param question: the question of a single line.
    :param answer: the answer of a single line.
    :param ground_truth: the ground truth of a single line.
    """

    azure_configs = {
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "model_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "model_name": os.getenv("AZURE_OPENAI_MODEL"),
        "embedding_deployment": os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        "embedding_name": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_MODEL")
    }

    azure_model = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    )
    
    azure_embeddings = AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    )

    data = {
        'question': [question],
        'answer': [answer],
        'ground_truth': [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    score = evaluate(dataset,metrics=[answer_correctness], llm=azure_model, embeddings=azure_embeddings)
    score.to_pandas()

    return score['answer_correctness']

    
