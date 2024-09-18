from configure_azure import configure_azure 
from data_enron import load_enron
from data_amnesty import load_amnesty

azure_model, azure_embeddings = configure_azure()
#transformed_df = load_enron()
transformed_df = load_amnesty()


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
