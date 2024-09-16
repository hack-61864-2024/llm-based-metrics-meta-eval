import datetime
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from llmops.src.common import GitRepositoryStatus
from llmops.src.run_evaluation_flow import EvaluatorSelector, run_evaluation_flow

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


def test_run_evaluation_flow():
    with patch("llmops.src.run_evaluation_flow.wait_job_finish"), patch(
        "llmops.src.run_evaluation_flow.get_variant_from_run"
    ) as mock_get_variant_from_run, patch("llmops.src.run_evaluation_flow.PFClient") as mock_pf_client, patch(
        "llmops.src.run_evaluation_flow.datetime"
    ) as mock_datetime, patch(
        "llmops.src.run_evaluation_flow.dotenv_values"
    ) as mock_dotenv_values:
        # Mock the PFClient
        pf_client_instance = Mock()
        mock_pf_client.return_value = pf_client_instance
        # Mock the standard run
        standard_run_source_0 = "ds1_source"
        standard_run_source_1 = "ds2_source"

        run_names = ["run_id_0", "run_id_1"]
        standard_run_instance_0 = Mock(data=standard_run_source_0)
        standard_run_instance_0.tags = []
        standard_run_instance_0.name = run_names[0]
        standard_run_instance_1 = Mock(data=standard_run_source_1)
        standard_run_instance_1.tags = []
        standard_run_instance_1.name = run_names[1]
        run_dict = {run_names[0]: standard_run_instance_0, run_names[1]: standard_run_instance_1}

        pf_client_instance.runs.get.side_effect = lambda run_id: run_dict[run_id]

        # Mock the run details and metrics
        pf_client_instance.get_metrics.return_value = {}
        pf_client_instance.get_details.return_value = pd.DataFrame()

        # Mock get_variant_from_run
        mock_get_variant_from_run.return_value = None

        # Mock dotenv_values
        mock_dotenv_values.return_value = {}

        # Mock the datetime
        fake_date = datetime.datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.datetime.now.return_value = fake_date

        # Start the run
        report_dir = tempfile.TemporaryDirectory()
        run_identity = Mock()
        run_evaluation_flow(
            run_id="['run_id_0', 'run_id_1']",
            exp_base_path=str(RESOURCE_PATH),
            report_dir=report_dir.name,
            run_identity=run_identity,
            evaluator_selector=EvaluatorSelector(selector=1, selected_evaluators=["eval1", "eval2", "3"]),
        )

        # Get the argument of each time pf_client.runs.create_or_update is called
        created_runs = pf_client_instance.runs.create_or_update

        # Expect 3 created runs
        # Two for the "eval1" evaluator, one using ds1_source and using for ds2_source
        # One for the "eval2" evaluator using ds2_source

        # Expected run arguments
        expected_run = [standard_run_instance_0, standard_run_instance_1, standard_run_instance_1]
        expected_data = [standard_run_source_0, standard_run_source_1, standard_run_source_1]
        expected_column_mappings = [
            {"ds1_input": "ds1_mapping", "ds1_extra": "ds1_extra_mapping"},
            {"ds2_extra": "ds2_extra_mapping"},
            {"ds2_input": "ds2_diff_mapping"},
        ]
        expected_env_vars = {"PF_LOGGING_LEVEL": "WARNING"}

        assert created_runs.call_count == len(expected_run)

        # created_runs.call_args_list is triple nested,
        # first index: select the call of pf_client_instance.runs.create_or_update [0, 5]
        # second index: select the argument of pf_client_instance.runs.create_or_update [0 (run), 1 (stream)]
        # third index: select the first element of the tuple [0]
        for i, call_args in enumerate(created_runs.call_args_list):
            run = call_args[0][0]
            assert run.run == expected_run[i]
            assert run.data == expected_data[i]
            assert run.column_mapping == expected_column_mappings[i]
            assert run.environment_variables == expected_env_vars
            assert run.tags["evaluator"] is not None
            assert run.tags["env_name"] == "local"
            assert run.tags["source_run"] in run_names
            assert run.tags["branch"] == TEST_BRANCH

        # Check that reports exist
        timestamp = fake_date.strftime("%Y%m%d_%H%M%S")
        report_subdir = os.path.join(report_dir.name, timestamp)
        assert os.path.exists(os.path.join(report_subdir, "eval1_metrics.csv"))
        assert os.path.exists(os.path.join(report_subdir, "eval1_metrics.html"))
        assert os.path.exists(os.path.join(report_subdir, "eval1_results.csv"))
        assert os.path.exists(os.path.join(report_subdir, "eval1_results.html"))
        assert os.path.exists(os.path.join(report_subdir, "eval2_metrics.csv"))
        assert os.path.exists(os.path.join(report_subdir, "eval2_metrics.html"))
        assert os.path.exists(os.path.join(report_subdir, "eval2_results.csv"))
        assert os.path.exists(os.path.join(report_subdir, "eval2_results.html"))
        report_dir.cleanup()
