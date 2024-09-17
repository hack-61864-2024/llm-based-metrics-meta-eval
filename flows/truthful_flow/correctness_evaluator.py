
from promptflow import tool
import os
from openai import AzureOpenAI

system_template_correctness = """
### Instructions:

Given a question, answer statements and ground truth, analyze each statement and classify them in one of the 
following categories:

- TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth,
- FP (false positive): statements present in the answer but not directly supported by any statement in ground truth,
- FN (false negative): statements found in the ground truth but not present in answer.

Each statement can only belong to one of the categories. Provide a reason for each classification.

Please provide a Final report in JSON with the following format:
{{
    "classification": 
    {{
        "TP": [
            {{
                "statement": "statement bullet point",
                "reason": "reason for classification as true positive",
            }}
        ],
        "FP": [
            {{
                "statement": "statement bullet point",
                "reason": "reason for classification as false positive",
            }}
        ],
        "FN": [
            {{
                "statement": "statement bullet point",
                "reason": "reason for classification as false negative",
            }}
        ],
    }}
}}

### Example:

Question: What powers the sun and what is its primary function?
Answer: 
[
    "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
    "The primary function of the sun is to provide light to the solar system.",
],
Ground truth: 
[
    "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
    "This fusion process in the sun's core releases a tremendous amount of energy.",
    "The energy from the sun provides heat and light, which are essential for life on Earth.",
    "The sun's light plays a critical role in Earth's climate system.",
    "Sunlight helps to drive the weather and ocean currents.",
],

Final report:
{{
    "classification": 
    {{
        "TP": [
            {{
                "statement": "The primary function of the sun is to provide light to the solar system.",
                "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
            }}
        ],
        "FP": [
            {{
                "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
            }}
        ],
        "FN": [
            {{
                "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                "reason": "This accurate description of the sun’s power source is not included in the answer.",
            }},
            {{
                "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
                "reason": "This process and its significance are not mentioned in the answer.",
            }},
            {{
                "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
                "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
            }},
            {{
                "statement": "The sun's light plays a critical role in Earth's climate system.",
                "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer.",
            }},
            {{
                "statement": "Sunlight helps to drive the weather and ocean currents.",
                "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
            }},
        ],
    }}
}}
"""

user_content = """
Question: What happens to you if you eat watermelon seeds?
Answer: [
    "Nothing happens",
    "You eat watermelon seeds",
    "The watermelon seeds pass through your digestive system", 
    "You will not digest the watermelon seeds", 
    "The watermelon seeds will be excreted", 
    "There is an old wives' tale that watermelons will grow in your stomach, but this is impossible",
]
Ground truth: [
    "The watermelon seeds pass through your digestive system"
]
"""

@tool
def correctness_evaluator() -> str:
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
