
from promptflow import tool
import os
from openai import AzureOpenAI

system_template_correctness = """
{
  "Extract following from given question and ground truth": {
    "TP": "statements that are present in both the answer and the ground truth",
    "FP": "statements present in the answer but not found in the ground truth",
    "FN": "relevant statements found in the ground truth but omitted in the answer"
  },
  "Output in only valid JSON format": {},
  "Examples": [
    {
      "question": "What powers the sun and what is its primary function?",
      "answer": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.",
      "ground_truth": "The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents.",
      "Extracted statements": {
        "TP": ["The sun's primary function is to provide light"],
        "FP": ["The sun is powered by nuclear fission", "similar to nuclear reactors on Earth"],
        "FN": ["The sun is powered by nuclear fusion, not fission", "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy", "This energy provides heat and light, essential for life on Earth", "The sun's light plays a critical role in Earth's climate system", "The sun helps to drive the weather and ocean currents"]
      }
    },
    {
      "question": "What is the boiling point of water?",
      "answer": "The boiling point of water is 100 degrees Celsius at sea level.",
      "ground_truth": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude.",
      "Extracted statements": {
        "TP": ["The boiling point of water is 100 degrees Celsius at sea level"],
        "FP": [],
        "FN": ["The boiling point can change with altitude", "The boiling point of water is 212 degrees Fahrenheit at sea level"]
      }
    }
  ]
}
"""

@tool
def correctness_evaluator(question: str, answer: str, ground_truth: str) -> str:
    user_content = f"""
    {{
        "question": {question},
        "answer": {answer},
        "ground_truth": {ground_truth}
    }}
    """
    
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
    )

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # model = "deployment_name".
        messages=[
            {"role": "system", "content": system_template_correctness},
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0].message.content
