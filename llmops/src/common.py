from __future__ import annotations

import ast
import logging
import os
import random
import re
import string
import subprocess
import time
from typing import Optional, Sequence, Union

from azure.identity import AzureCliCredential, DefaultAzureCredential
from promptflow.entities import Run

LOGGER_NAME = "llmops"

RUN_TAG_ENVIRONMENT = "env_name"
RUN_TAG_EVALUATOR_NAME = "evaluator"
RUN_TAG_BRANCH = "branch"
RUN_TAG_COMMIT_ID = "commit_id"
RUN_TAG_NAME_SUFFIXES = "name_suffixes"
RUN_TAG_SOURCE_RUN = "source_run"
RUN_TAG_DATASET = "dataset"
RUN_TAG_PULL_REQUEST_ID = "pull_request"

# Set up logger for all `llmops` modules
logging.basicConfig(
    format="%(asctime)s %(module)-20s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    return logger


def get_credentials(
    use_az_cli_credentials: bool,
) -> Union[AzureCliCredential, DefaultAzureCredential]:
    if use_az_cli_credentials:
        get_logger().info("Using Azure CLI credentials")
        return AzureCliCredential()

    get_logger().info("Using default Azure credentials")
    return DefaultAzureCredential()


def resolve_run_ids(run_id: str) -> list[str]:
    """
    Read run_id from string or from file.

    :param run_id: List of run IDs (example '["run_id_1", "run_id_2", ...]') OR path to file containing list of run IDs.
    :type run_id: str
    :return: List of run IDs.
    :rtype: List[str]
    """
    if os.path.isfile(run_id):
        with open(run_id, "r") as run_file:
            raw_runs_ids = run_file.read()
            run_ids = [] if raw_runs_ids is None else ast.literal_eval(raw_runs_ids)
    else:
        run_ids = [] if run_id is None else ast.literal_eval(run_id)

    return run_ids


def resolve_model_version(model_version: str) -> str:
    """
    Read model_version from string or from file.

    :param model_version: Model version (example '42') OR path to file containing the model version.
    :type model_version: str
    :return: Model version.
    :rtype: str
    """
    if os.path.isfile(model_version):
        with open(model_version, "r") as run_file:
            version = run_file.read()
    else:
        version = model_version

    return version


def resolve_exp_base_path(base_path: str) -> str:
    """
    Read base_path as absolute or relative path or raise an exception.

    :param base_path: Absolute path or relative path or None.
    :type base_path: str
    :return: Absolute path or None.
    :rtype: str
    :raises ValueError: If provided base_path is not None, not a valid absolute path and not a valid relative path.
    """
    exp_base_path = base_path
    if exp_base_path is not None:
        if not os.path.isabs(exp_base_path):
            exp_base_path = os.path.join(os.getcwd(), exp_base_path)
            if not os.path.isdir(exp_base_path):
                raise ValueError(f"Experiment base path '{base_path}' does not exist.")
        print(f"Experiment base path: {exp_base_path}")
    return exp_base_path


def create_run_tags(
    common_tags: dict[str, str],
    git_status: GitRepositoryStatus,
    name_suffixes: Optional[Sequence[str]] = None,
    node_id: Optional[str] = None,
    variant_id: Optional[str] = None,
    variants: Optional[dict[str, str]] = None,
    dataset_name: Optional[str] = None,
) -> dict[str, str]:
    """
    Create the run tags from a list of common tags and the node and variant ids.

    :param common_tags: Dictionary of common tags.
    :type common_tags: dict[str, str]
    :param git_status: Git repository status.
    :type git_status: GitRepositoryStatus
    :param node_id: Used llm node.
    :type node_id: Optional[str]
    :param variant_id: Used llm node variant.
    :type variant_id: Optional[str]
    :param variants: Dictionary of all available variants.
    :type variants: Optional[dict[str, str]]
    :param name_suffixes: Contains name suffixes to be added to display name.
    :type name_suffixes: Optional[Sequence[str]]
    :return: Dictionary of run tags.
    """
    run_tags = common_tags.copy()
    if variants is not None:
        for node_id, variant_id in variants.items():
            run_tags[node_id] = variant_id
    if node_id is not None and variant_id is not None:
        run_tags[node_id] = variant_id

    if RUN_TAG_BRANCH not in run_tags and git_status.branch:
        run_tags[RUN_TAG_BRANCH] = git_status.branch

    if RUN_TAG_COMMIT_ID not in run_tags and git_status.commit_id is not None:
        run_tags[RUN_TAG_COMMIT_ID] = git_status.commit_id

    if name_suffixes is not None:
        run_tags[RUN_TAG_NAME_SUFFIXES] = ",".join(name_suffixes)

    if RUN_TAG_PULL_REQUEST_ID not in run_tags:
        pull_request_id = os.environ.get(GitRepositoryStatus.GITHUB_REF_ENV_VAR, None)
        if pull_request_id:
            run_tags[RUN_TAG_PULL_REQUEST_ID] = pull_request_id.removesuffix("/merge")

    if dataset_name is not None:
        run_tags[RUN_TAG_DATASET] = dataset_name

    return run_tags


def wait_job_finish(job: Run):
    """
    Wait for job to complete/finish

    :param job: The prompt flow run object.
    :type job: Run
    :raises Exception: If job not finished after 3 attempts with 5 second wait.
    """
    max_tries = 3
    attempt = 0
    logger = get_logger()
    while attempt < max_tries:
        logger.info(
            "\nWaiting for job %s to finish (attempt: %s)...",
            job.name,
            str(attempt + 1),
        )
        time.sleep(5)
        if job.status in ["Completed", "Finished"]:
            return
        attempt = attempt + 1

    raise Exception("Sorry, exiting job with failure..")


class GitRepositoryStatus:
    """
    Holds the status of a git repository inside/outside an CI/CD pipeline.
    In Github Action, branch name and commit id are taken from environment variables.
    """

    GITHUB_SHA_ENV_VAR = "GITHUB_SHA"
    GITHUB_REF_ENV_VAR = "GITHUB_REF_NAME"
    GITHUB_PR_HEAD_REF_ENV_VAR = "GITHUB_HEAD_REF"

    def __init__(
        self,
        branch_name: Optional[str],
        commit_id: Optional[str],
        has_pending_changes: bool,
    ):
        # On a CI build, the branch name should be taken from environment variable
        logger = get_logger()

        # This block is just overriding the git status setup from local for CI run, as the standard git status gives
        # false information.
        # We cannot use the same environment variable for both the pull request and standard push events as
        # GITHUB_REF points to the merge branch in case of PR build (and we want the source branch)
        # GITHUB_HEAD_REF does not exist in case of a non PR run
        # More information on https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables
        if GitRepositoryStatus.GITHUB_PR_HEAD_REF_ENV_VAR in os.environ:
            # PULL REQUEST BUILD
            logger.info("Running in the context of a Github Action Pull Request pipeline")
            ci_branch_name = os.environ.get(GitRepositoryStatus.GITHUB_PR_HEAD_REF_ENV_VAR, None)
            if ci_branch_name:
                branch_name = ci_branch_name
                logger.info("Resolved branch name from environment variables: %s", branch_name)
            has_pending_changes = False
            ci_commit_id = os.environ.get(GitRepositoryStatus.GITHUB_SHA_ENV_VAR, None)
            if ci_commit_id:
                commit_id = ci_commit_id
                logger.info("Resolved commit id from environment variables: %s", commit_id)
                has_pending_changes = False

        self.branch = branch_name
        self.commit_id = commit_id
        self.commit_id_short = commit_id[0:7] if commit_id else None
        self.has_pending_changes = has_pending_changes

    @classmethod
    def from_git(cls) -> GitRepositoryStatus:
        try:
            # Currently using a subprocess to avoid adding a python dependency for git
            process = subprocess.run(
                "git branch --show-current && git rev-parse HEAD && git status --porcelain",
                capture_output=True,
                shell=True,
                check=True,
            )
            results = process.stdout.decode("utf-8").strip().split("\n")
            branch_name = results[0] if len(results) > 0 else None
            commit_id = results[1] if len(results) > 1 else None
            has_pending_changes = len(results) > 2 and len(results[2]) > 0  # noqa: PLR2004

            return cls(branch_name, commit_id, has_pending_changes)
        except Exception:
            get_logger().warning(
                "💡Could not get the current git branch name. Ensure your project is using GIT!",
                exc_info=True,
            )

        return cls(None, None, False)


class RunName:
    """
    Defines the Prompt Flow run name
    """

    def __init__(self, name: str, display_name: str):
        self.name = name
        self.display_name = display_name

    @staticmethod
    def _sanitize_branch_name(name: Optional[str]) -> Optional[str]:
        """
        Remove "feature/" prefix from the branch name.
        Replace non-alphanumeric and dash with a dash in branch name.
        Remove dash suffix.
        So:
            feature/AIPE-17000-experiment -> AIPE-17000-experiment
            experiment/3111-experiment_2! -> experiment-3111-experiment-2
        """
        if not name:
            return name

        name = name.strip().removeprefix("feature/")
        name = re.sub(r"[^a-zA-Z0-9-]", "-", name)
        return name.removesuffix("-")

    @staticmethod
    def _resolve_eval_branch_name(source_run: Run, eval_branch_name: Optional[str]) -> Optional[str]:
        """
        Resolves the branch name for an evaluation for naming purposes
        Priority:
        1. The branch of the source_run
        2. The current branch name
        """

        if source_run.tags:
            tags_list: list[dict[str, str]] = source_run.tags
            if isinstance(tags_list, dict):
                tags_list = [tags_list]

            for tags in tags_list:
                if RUN_TAG_BRANCH in tags:
                    return tags[RUN_TAG_BRANCH]

        return eval_branch_name

    @classmethod
    def for_standard_run(
        cls,
        has_multiple_datasets: bool,
        experiment_name: str,
        data_ref: str,
        git_status: GitRepositoryStatus,
        timestamp: str,
        variant_id: Optional[str] = None,
        suffixes: Optional[Sequence[str]] = None,
    ) -> "RunName":
        """
        Create a run name (aka run id) and display name for a standard run.
        """

        # Only append the data_ref if there are multiple datasets
        data_ref_value = ""
        data_ref_value_for_display_name = ""
        if has_multiple_datasets:
            data_ref_value = f"_{data_ref}"
            data_ref_value_for_display_name = f" dataset={data_ref}"

        variant_value_for_name = ""
        variant_value_for_display_name = ""
        if variant_id:
            variant_value_for_name = f"_{variant_id}"
            variant_value_for_display_name = f" variant={variant_id}"
        name_prefix = f"{experiment_name}_{RunName._sanitize_branch_name(git_status.branch)}"
        name = f"{name_prefix}{variant_value_for_name}{data_ref_value}_{timestamp}"

        display_name_prefix = f"{experiment_name}_{RunName._sanitize_branch_name(git_status.branch)}"
        suffixes_display_name_value = " " + " ".join(suffixes) if suffixes else ""
        display_name = f"{display_name_prefix}{variant_value_for_display_name}{data_ref_value_for_display_name}{suffixes_display_name_value}"

        return cls(name, display_name)

    @classmethod
    def for_evaluation_run(
        cls,
        has_multiple_datasets: bool,
        data_ref: str,
        eval_name: str,
        experiment_name: str,
        timestamp: str,
        source_run: Run,
        git_status: GitRepositoryStatus,
        suffixes: Optional[Sequence[str]],
        variant_id: Optional[str],
    ) -> "RunName":
        """
        Create a run name (aka run id) and display name for an evaluation run.
        """

        # Only append the data_ref if there are multiple datasets
        data_ref_value = ""
        data_ref_value_for_display_name = ""
        if has_multiple_datasets:
            data_ref_value = f"_{data_ref}"
            data_ref_value_for_display_name = f" dataset={data_ref}"

        branch_name = RunName._resolve_eval_branch_name(source_run, git_status.branch)
        variant_value = f"_{variant_id}" if variant_id else ""

        name_prefix = f"{eval_name}_{experiment_name}_{RunName._sanitize_branch_name(branch_name)}"
        name = f"{name_prefix}{variant_value}{data_ref_value}_{timestamp}"

        display_name_prefix = f"{eval_name}_{experiment_name}_{RunName._sanitize_branch_name(branch_name)}"
        variant_value_for_display_name = f" variant={variant_id}" if variant_id else ""
        suffixes_display_name_value = " " + " ".join(suffixes) if suffixes else ""
        display_name = f"{display_name_prefix}{variant_value_for_display_name}{data_ref_value_for_display_name}{suffixes_display_name_value}"

        return cls(name, display_name)


def generate_random_name_part() -> str:
    """
    Generate a random name containing 6 lowercase characters from the ascii alphabet.
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(6))


class RunIdentityConfig:
    """
    Configuration for the identity of a run.
    """

    def __init__(self, identity: Optional[str] = None, client_id: Optional[str] = None):
        # If we are running in the CI context, we want to run with the AML identity, therefore managed with no client_id
        # specified, see https://microsoft.github.io/promptflow/reference/run-yaml-schema-reference.html#run-with-identity-examples
        if os.getenv("CI"):
            self._identity = "managed"
            self._client_id = None
        else:
            self._identity = identity
            self._client_id = client_id

    def to_identity(self) -> dict[str, str]:
        if not self._identity:
            return {}
        result = {"type": self._identity}
        if self._client_id:
            result["client_id"] = self._client_id

        return result
