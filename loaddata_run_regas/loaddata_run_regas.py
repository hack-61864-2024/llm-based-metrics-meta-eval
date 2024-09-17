parquetpath = './test-00000-of-00001.parquet' # using data from https://huggingface.co/datasets/MichaelR207/enron_qa_0822  test-00000-of-00001.parquet

import os
import pandas as pd


#### - CONFIGURE  AZURE OPENAI - ####
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

load_dotenv()

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


#### - IMPORT THE DATA - ####
# Read the Parquet file
df = pd.read_parquet('test-00000-of-00001.parquet')
# fulldata = df.to_dict(orient='records') # DEBUG: Convert DataFrame to list of dictionaries (records format)


#### - CONVERT DATA TO REGAS EXPECTED INPUT - ####
transformed_data = []

# Iterate through each row in the original DataFrame
for _, row in df.iterrows():
    # Skip records with empty or null values in 'alternate_answers'
    if len(row['alternate_answers']) == 0 or len(row['alternate_answers'][0]) == 0:
        continue

    # Create the transformed row
    transformed_row = {
        'question': row['questions'][0],          # Map 'question' to 'questions'
        'answer': row['alternate_answers'][0][0], # Map 'answer' to the first 'alternate_answers'
        'contexts': [row['email']],              # Map 'contexts' to 'email'
        'ground_truth': (row['gold_answers'])[0]    # Map 'ground_truth' to 'gold_answers'
    }
    transformed_data.append(transformed_row)

# Create the transformed DataFrame
transformed_df = pd.DataFrame(transformed_data)



#### - RAGAS EVALUATION - ====
## Ragas imports
from datasets import Dataset 
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from ragas.evaluation import evaluate

number_of_records = 10
print("STARTING EVALUATION")
print(f"Evaluating {number_of_records} records")
dataset = Dataset.from_pandas(transformed_df.head(number_of_records))
metrics = [
    faithfulness,
    answer_relevancy,
    # context_recall,
    # context_precision,
]
result = evaluate(dataset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings)
print("EVALUATION COMPLETE")
print(result)
print(result.to_pandas())
