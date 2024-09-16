import os
import random
import string
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from llmops.src.common import GitRepositoryStatus
from llmops.src.run_standard_flow import DatasetSelector, VariantsSelector, run_flow

THIS_PATH = Path(__file__).parent
RESOURCE_PATH = THIS_PATH / "resources"

TEST_BRANCH = "test_branch"


@pytest.fixture(scope="class", autouse=True)
def set_current_git_branch():
    """
    Set the current git branch as response of
    """
    with patch("llmops.src.common.GitRepositoryStatus.from_git"), patch(
        "llmops.src.run_standard_flow.GitRepositoryStatus.from_git"
    ) as mock_get_branch_name:
        mock_get_branch_name.return_value = GitRepositoryStatus(
            branch_name=TEST_BRANCH, commit_id=None, has_pending_changes=True
        )
        yield mock_get_branch_name


@pytest.fixture(scope="class", autouse=True)
def set_empty_local_var():
    """
    Set the current git branch as response of
    """
    with patch("llmops.src.run_standard_flow.dotenv_values") as mock_env_values:
        mock_env_values.return_value = {}
        yield mock_env_values


@pytest.fixture(scope="class", autouse=True)
def _set_required_env_vars():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("SUBSCRIPTION_ID", "TEST_SUBSCRIPTION_ID")
        monkeypatch.setenv("RUNTIME", "TEST_RUNTIME")
        monkeypatch.setenv("RESOURCE_GROUP_NAME", "TEST_RESOURCE_GROUP_NAME")
        monkeypatch.setenv("WORKSPACE_NAME", "TEST_WORKSPACE_NAME")
        monkeypatch.delenv(GitRepositoryStatus.GITHUB_PR_HEAD_REF_ENV_VAR, raising=False)
        monkeypatch.delenv(GitRepositoryStatus.GITHUB_REF_ENV_VAR, raising=False)
        monkeypatch.delenv("CI", raising=False)

        yield monkeypatch


def random_string():
    return "".join(random.choices(string.ascii_lowercase, k=10))


def test_dataset_selector():
    dataset_selector = DatasetSelector.from_args("*")
    assert dataset_selector.is_dataset_enabled(random_string())

    dataset_selector = DatasetSelector.from_args("all")
    assert dataset_selector.is_dataset_enabled(random_string())

    dataset_name = random_string()
    dataset_selector = DatasetSelector.from_args(dataset_name)
    assert dataset_selector.is_dataset_enabled(dataset_name)
    assert not dataset_selector.is_dataset_enabled(random_string())


def test_variant_selector():
    random_node = random_string()
    random_variant = random_string()
    selected_variant = random_string()
    selected_node = random_string()

    variant_selector = VariantsSelector.from_args("*")
    assert variant_selector.is_variant_enabled(random_node, random_variant)

    variant_selector = VariantsSelector.from_args("all")
    assert variant_selector.is_variant_enabled(random_node, random_variant)

    # TO DO check that this is the intended behavior
    variant_selector = VariantsSelector.from_args("default")
    assert variant_selector.is_variant_enabled(random_node, random_variant)

    variant_selector = VariantsSelector.from_args("defaults")
    assert variant_selector.is_variant_enabled(random_node, random_variant)

    variant_selector = VariantsSelector.from_args(f"{selected_node}.{selected_variant}")
    assert variant_selector.is_variant_enabled(selected_node, selected_variant)
    assert not variant_selector.is_variant_enabled(random_node, random_variant)


def test_run_standard_flow_all():
    variant_selector = VariantsSelector.from_args("*")
    dataset_selector = DatasetSelector.from_args("*")
    with patch("llmops.src.run_standard_flow.wait_job_finish"), patch(
        "llmops.src.run_standard_flow.PFClient"
    ) as mock_pf_client:
        # Mock the PFClient
        pf_client_instance = Mock()
        mock_pf_client.return_value = pf_client_instance
        pf_client_instance.get_details.return_value = pd.DataFrame()
        run_identity = Mock()
        # Start the run
        run_flow(
            variants_selector=variant_selector,
            datasets_selector=dataset_selector,
            exp_base_path=str(RESOURCE_PATH),
            run_identity=run_identity,
        )

        # Get the argument of each time pf_client.runs.create_or_update is called
        created_runs = pf_client_instance.runs.create_or_update

        # Expect 6 created runs
        # {node_var_0.var_0; node_var_1.var_3; ds1}
        # {node_var_0.var_1; node_var_1.var_3; ds1}
        # {node_var_0.var_0; node_var_1.var_4; ds1}
        # {node_var_0.var_0; node_var_1.var_3; ds2}
        # {node_var_0.var_1; node_var_1.var_3; ds2}
        # {node_var_0.var_0; node_var_1.var_4; ds2}

        # Expected run arguments
        expected_variants = [
            "${node_var_0.var_0}",
            "${node_var_0.var_1}",
            "${node_var_1.var_4}",
            "${node_var_0.var_0}",
            "${node_var_0.var_1}",
            "${node_var_1.var_4}",
        ]
        expected_data = [
            "ds1_source",
            "ds1_source",
            "ds1_source",
            "ds2_source",
            "ds2_source",
            "ds2_source",
        ]
        expected_column_mappings = [
            {"ds1_input": "ds1_mapping"},
            {"ds1_input": "ds1_mapping"},
            {"ds1_input": "ds1_mapping"},
            {"ds2_input": "ds2_mapping"},
            {"ds2_input": "ds2_mapping"},
            {"ds2_input": "ds2_mapping"},
        ]
        expected_env_vars = {"PF_LOGGING_LEVEL": "WARNING"}

        assert created_runs.call_count == len(expected_variants)

        # created_runs.call_args_list is triple nested,
        # first index: select the call of pf_client_instance.runs.create_or_update [0, 5]
        # second index: select the argument of pf_client_instance.runs.create_or_update [0 (run), 1 (stream)]
        # third index: select the first element of the tuple [0]
        for i, call_args in enumerate(created_runs.call_args_list):
            run = call_args[0][0]
            assert run.variant == expected_variants[i]
            assert run.data == expected_data[i]
            assert run.column_mapping == expected_column_mappings[i]
            assert run.environment_variables == expected_env_vars
            assert "evaluator" not in run.tags
            assert run.tags["env_name"] == "local"
            assert run.tags["branch"] == TEST_BRANCH


def test_run_standard_flow_default():
    variant_selector = VariantsSelector.from_args("default")
    dataset_selector = DatasetSelector.from_args("*")
    with patch("llmops.src.run_standard_flow.wait_job_finish"), patch(
        "llmops.src.run_standard_flow.PFClient"
    ) as mock_pf_client:
        # Mock the PFClient
        pf_client_instance = Mock()
        mock_pf_client.return_value = pf_client_instance
        pf_client_instance.get_details.return_value = pd.DataFrame()
        run_identity = Mock()
        # Start the run
        run_flow(
            variants_selector=variant_selector,
            datasets_selector=dataset_selector,
            exp_base_path=str(RESOURCE_PATH),
            run_identity=run_identity,
        )

        # Get the argument of each time pf_client.runs.create_or_update is called
        created_runs = pf_client_instance.runs.create_or_update

        # Expect 2 created runs
        # {node_var_0.var_0; node_var_1.var_3; ds1}
        # {node_var_0.var_1; node_var_1.var_3; ds2}

        # Expected run arguments
        expected_data = [
            "ds1_source",
            "ds2_source",
        ]
        expected_column_mappings = [
            {"ds1_input": "ds1_mapping"},
            {"ds2_input": "ds2_mapping"},
        ]
        expected_env_vars = {"PF_LOGGING_LEVEL": "WARNING"}

        assert created_runs.call_count == len(expected_data)
        for i, call_args in enumerate(created_runs.call_args_list):
            run = call_args[0][0]
            assert run.variant is None  # Run will select the default variant
            assert run.data == expected_data[i]
            assert run.column_mapping == expected_column_mappings[i]
            assert run.environment_variables == expected_env_vars
            assert "evaluator" not in run.tags
            assert run.tags["env_name"] == "local"
            assert run.tags["branch"] == TEST_BRANCH


def test_run_standard_flow_custom():
    variant_selector = VariantsSelector.from_args("node_var_0.var_1, node_var_1.var_4")
    dataset_selector = DatasetSelector.from_args("*")
    with patch("llmops.src.run_standard_flow.wait_job_finish"), patch(
        "llmops.src.run_standard_flow.PFClient"
    ) as mock_pf_client:
        # Mock the PFClient
        pf_client_instance = Mock()
        mock_pf_client.return_value = pf_client_instance
        pf_client_instance.get_details.return_value = pd.DataFrame()
        run_identity = Mock()

        # Start the run
        run_flow(
            variants_selector=variant_selector,
            datasets_selector=dataset_selector,
            exp_base_path=str(RESOURCE_PATH),
            run_identity=run_identity,
        )

        # Get the argument of each time pf_client.runs.create_or_update is called
        created_runs = pf_client_instance.runs.create_or_update

        # Expect 4 created runs
        # {node_var_0.var_1; node_var_1.var_3; ds1}
        # {node_var_0.var_0; node_var_1.var_4; ds1}
        # {node_var_0.var_1; node_var_1.var_3; ds2}
        # {node_var_0.var_0; node_var_1.var_4; ds2}

        # Expected run arguments
        expected_variants = [
            "${node_var_0.var_1}",
            "${node_var_1.var_4}",
            "${node_var_0.var_1}",
            "${node_var_1.var_4}",
        ]
        expected_data = ["ds1_source", "ds1_source", "ds2_source", "ds2_source"]
        expected_column_mappings = [
            {"ds1_input": "ds1_mapping"},
            {"ds1_input": "ds1_mapping"},
            {"ds2_input": "ds2_mapping"},
            {"ds2_input": "ds2_mapping"},
        ]
        expected_env_vars = {"PF_LOGGING_LEVEL": "WARNING"}

        assert created_runs.call_count == len(expected_variants)
        for i, call_args in enumerate(created_runs.call_args_list):
            run = call_args[0][0]
            assert run.variant == expected_variants[i]
            assert run.data == expected_data[i]
            assert run.column_mapping == expected_column_mappings[i]
            assert run.environment_variables == expected_env_vars
            assert "evaluator" not in run.tags
            assert run.tags["env_name"] == "local"
            assert run.tags["branch"] == TEST_BRANCH


def test_run_standard_flow_write_results():
    variant_selector = VariantsSelector.from_args("node_var_0.var_1, node_var_1.var_4")
    dataset_selector = DatasetSelector.from_args("*")

    # experiment properties (from experiment.yaml files)
    experiment_name = "exp"
    ds1 = "ds1_source"
    ds2 = "ds2_source"
    var_1 = "var_1"
    var_2 = "var_4"
    col = "col"
    dummy_data = "dummydata"

    dummy_results = pd.DataFrame({col: [dummy_data]})

    fake_date_time = datetime(2024, 1, 1, 12, 0, 0)
    fake_date_time_str = fake_date_time.strftime("%Y%m%d_%H%M%S")

    with patch("llmops.src.run_standard_flow.wait_job_finish"), patch(
        "llmops.src.run_standard_flow.PFClient"
    ) as mock_pf_client, patch("llmops.src.run_standard_flow.datetime.datetime") as mock_time:
        # Mock the PFClient
        pf_client_instance = Mock()
        mock_pf_client.return_value = pf_client_instance
        pf_client_instance.get_details.return_value = dummy_results
        run_identity = Mock()
        mock_time.now.return_value = fake_date_time

        with tempfile.TemporaryDirectory() as temp_dir:
            expected_csv = os.path.join(temp_dir, fake_date_time_str, f"{experiment_name}_standard_flow_results.csv")
            expected_html = os.path.join(temp_dir, fake_date_time_str, f"{experiment_name}_standard_flow_results.html")

            # Start the run
            run_flow(
                variants_selector=variant_selector,
                datasets_selector=dataset_selector,
                exp_base_path=str(RESOURCE_PATH),
                run_identity=run_identity,
                report_dir=temp_dir,
            )

            # Check that files exists
            assert os.path.exists(expected_csv)
            assert os.path.exists(expected_html)

            # Check the contents of the csv file
            df = pd.read_csv(expected_csv)

            # Check column titles
            expected_columns = [col, "dataset", "run_id", "variant_id", "experiment_name"]
            assert list(df.columns)[1:] == expected_columns

            # Check column "col"
            assert (df[col] == dummy_data).all()

            # Check column "experiment_name"
            assert (df["experiment_name"] == experiment_name).all()

            # Check column "dataset"
            expected_datasets = [ds1, ds1, ds2, ds2]
            assert (df["dataset"] == expected_datasets).all()

            # Check column "variant_id"
            expected_variants = [var_1, var_2, var_1, var_2]
            assert (df["variant_id"] == expected_variants).all()
