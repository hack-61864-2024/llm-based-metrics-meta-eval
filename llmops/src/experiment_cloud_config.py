import os
from typing import Optional


def _try_get_env_var(variable_name: str) -> str:
    """
    Try to read environment variable. Raise ValueError if the variable doesn't exist or is empty.

    :param variable_name: Environment variable name.
    :type variable_name: str
    :return: Value of the environment variable.
    :rtype: str
    :raises ValueError: If the variable doesn't exist or is empty.
    """
    value = os.environ.get(variable_name)
    if value is None or value == "":
        raise ValueError(f"Environment variable '{variable_name}' is not set or is empty.")
    return value


def _get_optional_env_var(variable_name: str) -> Optional[str]:
    """
    Read environment variable. Return None if the variable doesn't exist or is empty.

    :param variable_name: Environment variable name.
    :type variable_name: str
    :return: Value of the environment variable or None.
    :rtype: str
    """
    value = os.environ.get(variable_name)
    if value is None or value == "":
        return None
    return value


class ExperimentCloudConfig:
    """
    Configuration for running an experiment in the cloud.

    :param subscription_id: Subscription ID of the Azure ML workspace.
    :type subscription_id: str
    :param resource_group_name: Resource group name of the Azure ML workspace.
    :type resource_group_name: str
    :param workspace_name: Name of the Azure ML workspace.
    :type workspace_name: str
    :param runtime: Prompt Flow runtime to be used. If empty, automatic runtime will be used.
    :type runtime: str
    :param serverless_instance_type: Serverless instance type (defaults to STANDARD_DS3_V2).
    :type serverless_instance_type: str
    :param compute_instance_name: Name fo the compute instance.
    :type compute_instance_name: str
    """

    def __init__(self, subscription_id: Optional[str] = None):
        self.subscription_id = subscription_id or _try_get_env_var("SUBSCRIPTION_ID")
        self.resource_group_name = _try_get_env_var("RESOURCE_GROUP_NAME")
        self.workspace_name = _try_get_env_var("WORKSPACE_NAME")
        self.runtime = _get_optional_env_var("RUNTIME")
        self.serverless_instance_type = _get_optional_env_var("SERVERLESS_INSTANCE_TYPE") or "STANDARD_DS3_V2"
        self.compute_instance_name = _get_optional_env_var("COMPUTE_INSTANCE")
