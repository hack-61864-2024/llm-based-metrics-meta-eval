
import os
from openai import AzureOpenAI
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def get_answer_correctness_report(question: str, answer: str, ground_truth: list) -> str:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
    )

    prompt = """
    Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories: 

        - TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth,
        - FP (false positive): statements present in the answer but not directly supported by any statement in ground truth,
        - FN (false negative): statements found in the ground truth but not present in answer.

    Each statement can only belong to one of the categories. Provide a reason for each classification.
    """

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # model = "deployment_name".
        messages=[
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"question: {question}, answer: {answer}, ground truth: {ground_truth}" },
        ],
    )

    return response.choices[0].message.content
