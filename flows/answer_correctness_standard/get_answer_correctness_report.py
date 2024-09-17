
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
    You are an evaluator that takes a question, answer and ground truth data set from the user and evaluates it by comparing the ground truth and answer statements, classify them in one of the following categories: 

        - TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth,
        - FP (false positive): statements present in the answer but not directly supported by any statement in ground truth,
        - FN (false negative): statements found in the ground truth but not present in answer.

    Each statement can only belong to one of the categories. Provide a reason for each classification. 
    
    Here is an example of what the reponse format should be. Response should ONLY include json as formatted below:
        {
            "TP": [
                {
                    "statement": "The primary function of the sun is to provide light to the solar system.",
                    "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
                }
            ],
            "FP": [
                {
                    "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                    "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
                },
                {
                    "statement": "The sun is the center of the solar system.",
                    "reason": "This statement is extraneous and not included in the ground truth.",
                }
            ],
            "FN": [
                {
                    "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                    "reason": "This accurate description of the sun’s power source is not included in the answer.",
                },
                {
                    "statement": N/a,
                    "reason": "The answer does not mention this part of the the ground truth.",
                },
                {
                    "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
                },
                {
                    "statement": "The sun's light plays a critical role in Earth's climate system.",
                    "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer.",
                },
                {
                    "statement": N/A,
                    "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
                },
            ],
        }
    """

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # model = "deployment_name".
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Please evaluate the following data set: question: {question}, answer: {answer}, ground truth: {ground_truth}" },
        ],
    )

    return response.choices[0].message.content
