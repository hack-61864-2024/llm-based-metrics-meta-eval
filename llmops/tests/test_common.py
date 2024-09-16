import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from llmops.src.common import (
    GitRepositoryStatus,
    RunName,
    resolve_exp_base_path,
    resolve_model_version,
    resolve_run_ids,
)

THIS_PATH = Path(__file__).parent
RESOURCE_PATH = THIS_PATH / "resources"


@pytest.fixture(scope="class", autouse=True)
def _set_required_env_vars():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv(GitRepositoryStatus.GITHUB_PR_HEAD_REF_ENV_VAR, raising=False)
        monkeypatch.delenv(GitRepositoryStatus.GITHUB_REF_ENV_VAR, raising=False)

        yield monkeypatch


def test_resolve_exp_base_path():
    assert resolve_exp_base_path(None) is None
    assert resolve_exp_base_path(str(RESOURCE_PATH)) == os.path.join(os.getcwd(), str(RESOURCE_PATH))

    bad_path = "experiments"
    with pytest.raises(ValueError, match=f"Experiment base path '{bad_path}' does not exist."):
        resolve_exp_base_path(bad_path)


def test_resolve_model_version():
    assert resolve_model_version(str(THIS_PATH / "resources/model_version.txt")) == "1"
    assert resolve_model_version("3") == "3"


def test_resolve_run_ids():
    assert resolve_run_ids(str(THIS_PATH / "resources/run_ids.txt")) == ["run1", "run2"]
    assert resolve_run_ids('["run3"]') == ["run3"]


def test_get_branch_name():
    with patch("llmops.src.common.subprocess.run") as mock_popen:
        # ARRANGE
        process = MagicMock()
        process.configure_mock(**{"stdout.decode.return_value": "my_branch_name\ncommit_id\ntrue"})
        mock_popen.return_value = process

        # ACT
        result = GitRepositoryStatus.from_git()
        # ASSERT
        assert result.branch == "my_branch_name"
        assert result.commit_id == "commit_id"
        assert result.has_pending_changes
        mock_popen.assert_called_once_with(
            "git branch --show-current && git rev-parse HEAD && git status --porcelain",
            capture_output=True,
            shell=True,
            check=True,
        )


@pytest.mark.parametrize(
    ("branch_name", "sanitized_branch_name"),
    [
        ("feature/AIPE-17000-experiment", "AIPE-17000-experiment"),
        ("experiment/3111-experiment_2!", "experiment-3111-experiment-2"),
    ],
)
def test_sanitized_branch_name(branch_name: str, sanitized_branch_name: str):
    assert RunName._sanitize_branch_name(branch_name) == sanitized_branch_name  # noqa: SLF001


@pytest.mark.parametrize(
    ("source_run_tags", "branch_name", "resolved_branch_name"),
    [
        ({"branch": "branch_0"}, "branch_1", "branch_0"),
        ({}, "branch_1", "branch_1"),
    ],
)
def test_eval_branch_name(source_run_tags: dict, branch_name: str, resolved_branch_name: str):
    run = Mock(tags=source_run_tags)
    assert RunName._resolve_eval_branch_name(run, branch_name) == resolved_branch_name  # noqa: SLF001


@pytest.mark.parametrize(
    (
        "multi_datasets",
        "experiment_name",
        "data_ref",
        "branch_name",
        "timestamp",
        "variant_id",
        "suffixes",
        "expected_name",
        "expected_display_name",
    ),
    [
        (
            False,
            "experiment_name",
            "data_ref",
            "branch_name",
            "timestamp",
            None,
            None,
            "experiment_name_branch-name_timestamp",
            "experiment_name_branch-name",
        ),
        (
            True,
            "experiment_name",
            "data_ref",
            "branch_name",
            "timestamp",
            None,
            None,
            "experiment_name_branch-name_data_ref_timestamp",
            "experiment_name_branch-name dataset=data_ref",
        ),
        (
            True,
            "experiment_name",
            "data_ref",
            "branch_name",
            "timestamp",
            "variant_id",
            None,
            "experiment_name_branch-name_variant_id_data_ref_timestamp",
            "experiment_name_branch-name variant=variant_id dataset=data_ref",
        ),
        (
            True,
            "experiment_name",
            "data_ref",
            "branch_name",
            "timestamp",
            "variant_id",
            ["suffix_a", "suffix_b"],
            "experiment_name_branch-name_variant_id_data_ref_timestamp",
            "experiment_name_branch-name variant=variant_id dataset=data_ref suffix_a suffix_b",
        ),
        (
            False,
            "experiment_name",
            "data_ref",
            "branch_name",
            "timestamp",
            "variant_id",
            ["suffix_a", "suffix_b"],
            "experiment_name_branch-name_variant_id_timestamp",
            "experiment_name_branch-name variant=variant_id suffix_a suffix_b",
        ),
    ],
    ids=[
        "single dataset; no variant; no suffixes",
        "multi dataset; no variant; no suffixes",
        "multi dataset; variant; no suffixes",
        "multi dataset; variant; suffixes",
        "single dataset; variant; suffixes",
    ],
)
def test_standard_run_name(
    multi_datasets: bool,
    experiment_name: str,
    data_ref: str,
    branch_name: str,
    timestamp: str,
    variant_id: str,
    suffixes: list[str],
    expected_name: str,
    expected_display_name: str,
):
    run_name = RunName.for_standard_run(
        multi_datasets,
        experiment_name,
        data_ref,
        GitRepositoryStatus(branch_name, None, False),
        timestamp,
        variant_id,
        suffixes,
    )
    assert run_name.name == expected_name
    assert run_name.display_name == expected_display_name


@pytest.mark.parametrize(
    (
        "multi_datasets",
        "data_ref",
        "eval_name",
        "experiment_name",
        "timestamp",
        "source_run_tags",
        "branch_name",
        "suffixes",
        "variant_id",
        "expected_name",
        "expected_display_name",
    ),
    [
        (
            True,
            "data_ref",
            "evaluation_name",
            "experiment_name",
            "timestamp",
            {"branch": "branch_a"},
            "branch_b",
            None,
            None,
            "evaluation_name_experiment_name_branch-a_data_ref_timestamp",
            "evaluation_name_experiment_name_branch-a dataset=data_ref",
        ),
        (
            False,
            "data_ref",
            "evaluation_name",
            "experiment_name",
            "timestamp",
            {},
            "branch_b",
            None,
            None,
            "evaluation_name_experiment_name_branch-b_timestamp",
            "evaluation_name_experiment_name_branch-b",
        ),
        (
            False,
            "data_ref",
            "evaluation_name",
            "experiment_name",
            "timestamp",
            {},
            "branch_b",
            None,
            "variant_id",
            "evaluation_name_experiment_name_branch-b_variant_id_timestamp",
            "evaluation_name_experiment_name_branch-b variant=variant_id",
        ),
        (
            False,
            "data_ref",
            "evaluation_name",
            "experiment_name",
            "timestamp",
            {"branch": "branch_a"},
            "branch_b",
            ["suffix_a"],
            "variant_id",
            "evaluation_name_experiment_name_branch-a_variant_id_timestamp",
            "evaluation_name_experiment_name_branch-a variant=variant_id suffix_a",
        ),
    ],
    ids=[
        "source run branch; multi dataset; no variant; no suffixes",
        "git branch; single dataset; no variant; no suffixes",
        "git branch; single dataset; variant; no suffixes",
        "source run branch; single dataset; variant; suffixes",
    ],
)
def test_evaluation_run_name(
    multi_datasets: bool,
    data_ref: str,
    eval_name: str,
    experiment_name: str,
    timestamp: str,
    source_run_tags: dict[str, str],
    branch_name: str,
    suffixes: list[str],
    variant_id: str,
    expected_name: str,
    expected_display_name: str,
):
    run_name = RunName.for_evaluation_run(
        multi_datasets,
        data_ref,
        eval_name,
        experiment_name,
        timestamp,
        Mock(tags=source_run_tags),
        GitRepositoryStatus(branch_name, None, False),
        suffixes,
        variant_id,
    )
    assert run_name.name == expected_name
    assert run_name.display_name == expected_display_name
