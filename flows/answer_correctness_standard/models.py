from dataclasses import dataclass
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

@dataclass
class ModelData:
    name: str
    endpoint: str
    model_name: str
    api_version: str

load_dotenv()
credential = DefaultAzureCredential()
client = CognitiveServicesManagementClient(credential, subscription_id=os.getenv("SUBSCRIPTION_ID"))
Models = None

def get_models():
    global Models
    if Models is not None:
        return Models
    account_name = os.getenv("AZURE_OPENAI_ACCOUNT_NAME")
    deployments = client.deployments.list(os.getenv("RESOURCE_GROUP_NAME"), account_name)
    Models = {}
    for deployment in deployments:
        version = "2023-03-15-preview"
        model = ModelData(
            name = deployment.name,
            endpoint = f"https://{account_name}.openai.azure.com/openai/deployments/{deployment.name}/chat/completions?api-version={version}",
            model_name = deployment.properties.model.name,
            api_version = version,
        )
        Models[model.name] = model
    return Models

if __name__ == "__main__":
    load_dotenv()
    print(get_models())