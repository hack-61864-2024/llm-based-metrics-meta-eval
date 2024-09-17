
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
    {
    "Extract the following from the given question, answer and ground truth": {
        "TP": "statements that are present in both the answer and the ground truth",
        "FP": "statements present in the answer but not found in the ground truth",
        "FN": "relevant statements found in the ground truth but omitted in the answer"
    },
    "Output in only valid JSON format": {},
    "Output Examples": [
        {
            {
                "TP": [{"answer": "<answer statement>", "reason": "<reason for categorization>"}],
                "FP": [{"answer": "<answer statement>", "reason": "<reason for categorization>"}],
                "FN": [{"answer": "<answer statement>", "reason": "<reason for categorization>"}]
            },
            {
                "TP": [{"answer": "<answer statement>", "reason": "<reason for categorization>"}],
                "FP": [],
                "FN": [{"answer": "<answer statement>", "reason": "<reason for categorization>"}]
            }
        }
    ]
    """

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # model = "deployment_name".
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"question: {question}, answer: {answer}, ground truth: {ground_truth}" },
        ],
    )

    return response.choices[0].message.content
