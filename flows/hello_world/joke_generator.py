from promptflow import tool
import os
from openai import AzureOpenAI


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def joke_generator(joke_topic: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-15-preview",
    )

    response = client.chat.completions.create(
        model="gpt-35-turbo",  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a joke generator."},
            {"role": "user", "content": "Tell me a joke about " + joke_topic},
        ],
    )

    return response.choices[0].message.content
