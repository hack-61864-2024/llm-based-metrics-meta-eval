import pytest

from llmops.src.experiment_cloud_config import ExperimentCloudConfig, _get_optional_env_var, _try_get_env_var


def test_try_get_env_var(monkeypatch: pytest.MonkeyPatch):
    env_var_key = "TEST_ENV_VAR"
    env_var_val = "test_value"

    monkeypatch.setenv(env_var_key, env_var_val)
    assert _try_get_env_var(env_var_key) == env_var_val

    monkeypatch.setenv(env_var_key, "")
    with pytest.raises(ValueError, match=f"Environment variable '{env_var_key}' is not set or is empty."):
        _try_get_env_var(env_var_key)

    monkeypatch.delenv(env_var_key)
    with pytest.raises(ValueError, match=f"Environment variable '{env_var_key}' is not set or is empty."):
        _try_get_env_var(env_var_key)


def test_get_optional_env_var(monkeypatch: pytest.MonkeyPatch):
    env_var_key = "TEST_ENV_VAR"
    env_var_val = "test_value"

    monkeypatch.setenv(env_var_key, env_var_val)
    assert _get_optional_env_var(env_var_key) == env_var_val

    monkeypatch.setenv(env_var_key, "")
    assert _get_optional_env_var(env_var_key) is None

    monkeypatch.delenv(env_var_key)
    assert _get_optional_env_var(env_var_key) is None


def test_experiment_cloud_config(monkeypatch: pytest.MonkeyPatch):
    subscription_id = "subscription_id"
    rg_name = "rg_name"
    ws_name = "ws_name"
    runtime = "runtime"
    env_name = "env_name"

    # Check fails without RESOURCE_GROUP_NAME
    monkeypatch.setenv("SUBSCRIPTION_ID", subscription_id)
    monkeypatch.setenv("RESOURCE_GROUP_NAME", "")
    with pytest.raises(ValueError, match="Environment variable 'RESOURCE_GROUP_NAME' is not set or is empty."):
        ExperimentCloudConfig()

    # Check fails without WORKSPACE_NAME
    monkeypatch.setenv("RESOURCE_GROUP_NAME", rg_name)
    monkeypatch.setenv("WORKSPACE_NAME", "")
    with pytest.raises(ValueError, match="Environment variable 'WORKSPACE_NAME' is not set or is empty."):
        ExperimentCloudConfig()

    # Check fails without SUBSCRIPTION_ID
    monkeypatch.setenv("WORKSPACE_NAME", ws_name)
    monkeypatch.setenv("SUBSCRIPTION_ID", "")
    with pytest.raises(ValueError, match="Environment variable 'SUBSCRIPTION_ID' is not set or is empty."):
        ExperimentCloudConfig()

    # Check works with subscription_id argument
    monkeypatch.setenv("RUNTIME", "")
    monkeypatch.setenv("ENV_NAME", "")
    exp = ExperimentCloudConfig(subscription_id)

    assert exp.subscription_id == subscription_id
    assert exp.resource_group_name == rg_name
    assert exp.workspace_name == ws_name
    assert exp.runtime is None

    # Check works with subscription_id, runtime and env_name environment variables
    modified_subscription_id = "modified_subscription_id"
    monkeypatch.setenv("SUBSCRIPTION_ID", modified_subscription_id)
    monkeypatch.setenv("RUNTIME", runtime)
    monkeypatch.setenv("ENV_NAME", env_name)
    exp = ExperimentCloudConfig()

    assert exp.subscription_id == modified_subscription_id
    assert exp.resource_group_name == rg_name
    assert exp.workspace_name == ws_name
    assert exp.runtime == runtime
