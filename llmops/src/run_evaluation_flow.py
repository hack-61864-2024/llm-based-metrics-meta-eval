import argparse
import ast
import datetime
import json
import os
from enum import Enum
from typing import Optional

import pandas as pd
from dotenv import dotenv_values, load_dotenv
from promptflow.azure import PFClient
from promptflow.entities import Run

from llmops.src.common import (
    RUN_TAG_ENVIRONMENT,
    RUN_TAG_EVALUATOR_NAME,
    RUN_TAG_NAME_SUFFIXES,
    RUN_TAG_SOURCE_RUN,
    GitRepositoryStatus,
    RunIdentityConfig,
    RunName,
    create_run_tags,
    get_credentials,
    get_logger,
    resolve_exp_base_path,
    resolve_run_ids,
    wait_job_finish,
)
from llmops.src.experiment import load_experiment
from llmops.src.experiment_cloud_config import ExperimentCloudConfig

logger = get_logger()


class EvaluatorSelector:
    """
    Selects the evaluators to run. Options are all or custom.
    """

    class EvaluatorSelectionOption(Enum):
        ALL = 1
        CUSTOM = 2

    def __init__(
        self,
        selector: EvaluatorSelectionOption,
        selected_evaluators: Optional[list[str]] = None,
    ):
        self._selector = selector
        self._selected_evaluators = selected_evaluators or []

    def is_evaluator_enabled(self, evaluator: str) -> bool:
        return (
            self._selector == EvaluatorSelector.EvaluatorSelectionOption.ALL or evaluator in self._selected_evaluators
        )

    @classmethod
    def from_args(cls, evaluators: str) -> "EvaluatorSelector":
        evaluators = evaluators.strip().lower()
        if evaluators in ["*", "all"]:
            return cls(cls.EvaluatorSelectionOption.ALL)
        return cls(
            cls.EvaluatorSelectionOption.CUSTOM,
            [v.strip() for v in evaluators.split(",")],
        )

    @classmethod
    def all(cls) -> "EvaluatorSelector":
        return cls(cls.EvaluatorSelectionOption.ALL)


class Variant:
    """
    Defines a variant of an llm node.

    :param node_id: The llm node name.
    :type node_id: str
    :param variant_id: The llm variant name.
    :type variant_id: str
    """

    def __init__(self, node_id: str, variant_id: str):
        self.node_id = node_id
        self.variant_id = variant_id


def get_variant_from_run(run: Run) -> Optional[Variant]:
    """
    Get the node variant used in a run if available as a property.

    :param run: The prompt flow run object.
    :type run: Run
    """
    # TODO check if this is a default property
    variant_value = run.properties.get("azureml.promptflow.node_variant", None)
    if variant_value is None:
        return None

    start_index = variant_value.find("{") + 1
    end_index = variant_value.find("}")
    variant_value = variant_value[start_index:end_index].split(".", 1)
    return Variant(variant_value[0], variant_value[1])


def _resolve_name_suffixes(source_run: Run) -> Optional[list[str]]:
    """
    Get the name suffixes used in a run if available as a property.

    :param run: The prompt flow run object.
    :type run: Run
    """
    if source_run.tags:
        tags_list: list[dict[str, str]] = source_run.tags
        if isinstance(tags_list, dict):
            tags_list = [tags_list]

        for tags in tags_list:
            if RUN_TAG_NAME_SUFFIXES in tags:
                name_suffixes = tags[RUN_TAG_NAME_SUFFIXES]
                return name_suffixes.split(",")
    return None


def run_evaluation_flow(  # noqa C901
    run_id: str,
    evaluator_selector: EvaluatorSelector,
    run_identity: RunIdentityConfig,
    exp_filename: Optional[str] = None,
    exp_base_path: Optional[str] = None,
    report_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    tags: Optional[str] = None,
    subscription_id: Optional[str] = None,
    environment: Optional[str] = None,
    force_az_cli_credentials: bool = False,
):
    """
    Run all the evaluation flows of the experiment on all provided run_ids. If report_dir is not None, save the results
    and metrics of each evaluation run in csv and html format.
    WORKSPACE_NAME, RESOURCE_GROUP_NAME expected as environment variables. SUBSCRIPTION_ID expected as environment
    variable or argument.

    :param run_id: List of run IDs (example '["run_id_1", "run_id_2", ...]') OR path to file containing list of run IDs.
    :type run_id: str
    :param exp_filename: Name of the yaml file defining the experiment. Defaults to None, in which case
    "experiment.yaml" is used.
    :type exp_filename: Optional[str]
    :param exp_base_path: Path to the yaml file defining the experiment. Defaults to None, in which case current
    working directory is used.
    :type exp_base_path: Optional[str]
    :param report_dir: Path to folder in which to write the results and metrics of the evaluation flows.
    Defaults to None, in which case the results and metrics are not saved.
    :type report_dir: Optional[str]
    :param output_file: Path to file to write the version of the run IDs of the evaluation flows.
    Defaults to None, in which case the run IDs are not saved.
    :type output_file: Optional[str]
    :param tags: String reprisenting the dictionary of tags to be applied to the evaluation run.
    :type tags: Optional[str]
    :param subscription_id: Subscription ID, overwrites the SUBSCRIPTION_ID environment variable.
    Defaults to None.
    :type subscription_id: Optional[str]
    :param environment: Used environment ('dev', 'prd' etc.). Defaults to None.
    :type environment: Optional[str]
    :param force_az_cli_credentials: Force the usage of the Az CLI credentials. Default value is False,
    in which case DefaultAzureCredential() is used.
    :type force_az_cli_credentials: bool
    """
    # Load all necessary variables
    config = ExperimentCloudConfig(subscription_id)
    if not environment:
        # In CI we want to load the content of the environment variable if environment is not set
        environment = "dev" if os.getenv("CI") else "local"
    experiment = load_experiment(filename=exp_filename, base_path=exp_base_path, env=environment)
    experiment_name = experiment.name
    common_tags = {} if not tags else ast.literal_eval(tags)
    common_tags[RUN_TAG_ENVIRONMENT] = environment

    # Run_Id of standard runs for each of the datasets - e2e and fallback.
    run_ids = resolve_run_ids(run_id)
    if run_ids is None or len(run_ids) == 0:
        raise ValueError("No run ids found.")

    eval_flows = experiment.evaluators
    if eval_flows is None or len(eval_flows) == 0:
        raise ValueError("No evaluation flows found.")

    # Initialize clients
    pf = PFClient(
        get_credentials(force_az_cli_credentials),
        config.subscription_id,
        config.resource_group_name,
        config.workspace_name,
    )

    standard_flow_detail = experiment.get_flow_detail()
    default_variants = standard_flow_detail.default_variants

    eval_run_ids = []

    runs: dict[str, Run] = {}
    for run in run_ids:
        runs[run] = pf.runs.get(run)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    git_status = GitRepositoryStatus.from_git()

    identity = run_identity.to_identity()
    logger.info("Using identity %s", identity)

    for evaluator in eval_flows:
        if not evaluator_selector.is_evaluator_enabled(evaluator.name):
            logger.info("Skipping evaluator '%s'", evaluator.name)
            continue

        logger.info("Starting evaluation of '%s'", evaluator.name)

        dataframes = []
        metrics = []

        flow_name = evaluator.name
        env_vars = dotenv_values()
        env_vars["PF_LOGGING_LEVEL"] = "WARNING"

        evaluator_executed = False

        # Iterate over standard flow runs
        for flow_run in run_ids:
            logger.info("Preparing run '%s'", flow_run)

            # Get the evalautor mapping of the dataset used in the standard run
            # Skip the evaluation of this run if not found

            current_standard_run = runs[flow_run]
            run_data_id = current_standard_run.data
            if not run_data_id:
                logger.error("Run %s does not have a data reference", flow_run)
                raise ValueError(f"Run {flow_run} does not have a data reference.")
            dataset_mapping = evaluator.find_dataset_mapping(run_data_id)
            if dataset_mapping is None:
                continue

            # Create run tags
            column_mapping = dataset_mapping.mappings
            dataset = dataset_mapping.dataset
            data_id = dataset.source
            dataset_name = dataset.get_friendly_name()

            variant = get_variant_from_run(current_standard_run)
            run_tags = {}
            if variant is not None:
                run_tags = create_run_tags(
                    common_tags=common_tags,
                    git_status=git_status,
                    node_id=variant.node_id if variant is not None else None,
                    variant_id=variant.variant_id if variant is not None else None,
                    dataset_name=dataset_name,
                )
            else:
                run_tags = create_run_tags(
                    common_tags=common_tags,
                    variants=default_variants,
                    git_status=git_status,
                    dataset_name=dataset_name,
                )

            run_tags[RUN_TAG_EVALUATOR_NAME] = evaluator.name
            run_tags[RUN_TAG_SOURCE_RUN] = current_standard_run.name

            name = RunName.for_evaluation_run(
                has_multiple_datasets=len(experiment.datasets) > 1,
                data_ref=dataset.get_friendly_name(),
                eval_name=flow_name,
                experiment_name=experiment_name,
                timestamp=timestamp,
                source_run=current_standard_run,
                git_status=git_status,
                suffixes=_resolve_name_suffixes(current_standard_run),
                variant_id=variant.variant_id if variant is not None else None,
            )

            evaluator_executed = True
            # Create run object
            if not config.runtime:
                logger.info("Using automatic runtime and serverless compute for the prompt flow run")
            else:
                logger.info("Using runtime '%s' for the prompt flow run", config.runtime)

            run = Run(
                flow=evaluator.path,
                data=data_id,
                run=current_standard_run,
                name=name.name,
                display_name=name.display_name,
                environment_variables=env_vars,
                column_mapping=column_mapping,
                tags=run_tags,  # type: ignore
                runtime=config.runtime,
                resources=None if config.runtime else {"instance_type": config.serverless_instance_type},
                identity=identity,
            )
            run._experiment_name = experiment_name

            # Execute the run
            logger.info(
                "Starting prompt flow run '%s' in Azure ML. This can take a few minutes.",
                run.name,
            )
            # Launch the configured run in Azure ML.
            eval_job = pf.runs.create_or_update(run, stream=True)
            eval_run_ids.append(eval_job.name)
            wait_job_finish(eval_job)

            # Get run results and metrics of the evaluation run
            df_result = pf.get_details(eval_job, all_results=True)
            df_result["dataset"] = data_id
            df_result["source_run"] = flow_run
            df_result["exp_run"] = run.name

            metric_variant = pf.get_metrics(eval_job)
            metric_variant["dataset"] = data_id
            metric_variant["source_run"] = flow_run
            metric_variant["exp_run"] = run.name

            # Add variant information to results and metrics
            if variant is not None:
                df_result[variant.node_id] = variant.variant_id
                metric_variant[variant.node_id] = variant.variant_id

                for node_id, variant_id in default_variants.items():
                    if node_id != variant.node_id:
                        df_result[node_id] = variant_id
                        metric_variant[node_id] = variant_id

            dataframes.append(df_result.copy())
            metrics.append(metric_variant)

            logger.info("Run %s completed with status %s\n", run.name, run.status)
            logger.info("Results: \n%s", df_result.head(10))
            logger.info("Metrics: \n%s", json.dumps(metrics, indent=4))

        # Metrics and results are saved once for each evaluator
        # If the evaluator is running on multiple datasets or standard runs, the results of all
        # datasets and standard runs will be collected in the same file
        if evaluator_executed and report_dir:
            logger.info("Writing %s reports to '%s'", flow_name, report_dir)

            timestamped_report_dir = os.path.join(report_dir, timestamp)
            os.makedirs(timestamped_report_dir, exist_ok=True)

            # Write the results of the evaluation flow (all datasets/standard runs) into csv and html files
            results_df = pd.concat(dataframes, ignore_index=True)
            results_df["flow_name"] = flow_name

            results_df.to_csv(f"{timestamped_report_dir}/{flow_name}_results.csv")
            with open(
                f"{timestamped_report_dir}/{flow_name}_results.html",
                "w",
                encoding="utf-8",
            ) as results_html:
                results_html.write(results_df.to_html(index=False))

            # Write the metrics of the evaluation flow (all datasets/standard runs) into csv and html files
            metrics_df = pd.DataFrame(metrics)
            metrics_df["flow_name"] = flow_name

            metrics_df.to_csv(f"{timestamped_report_dir}/{flow_name}_metrics.csv")
            with open(
                f"{timestamped_report_dir}/{flow_name}_metrics.html",
                "w",
                encoding="utf-8",
            ) as metrics_html:
                metrics_html.write(metrics_df.to_html(index=False))

        logger.info("Finished evaluation of '%s'\n", evaluator.name)

    logger.info("Completed runs: %s", str(eval_run_ids))
    if output_file is not None:
        # Write the evaluation run ids to file
        with open(output_file, "w") as out_file:
            out_file.write(str(eval_run_ids))


def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser("run_evaluation_flow")
    parser.add_argument(
        "--file",
        type=str,
        help="The path to the experiment file. Default is 'experiment.yaml'",
        required=False,
        default="experiment.yaml",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ids of runs to be evaluated (File or comma separated string)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="A folder to save evaluation results and metrics",
    )
    parser.add_argument("--output-file", type=str, required=False, help="A file to save run ids")
    parser.add_argument(
        "--tags",
        type=str,
        help="Dictionary of tags to be added to the created Azure ML jobs",
        default=None,
    )
    parser.add_argument(
        "--base-path",
        type=str,
        help="Base path for experiment. Default is current working directory.",
        default=None,
    )
    parser.add_argument(
        "--subscription-id",
        type=str,
        help="Subscription ID, overwrites the SUBSCRIPTION_ID environment variable",
        default=None,
    )
    parser.add_argument(
        "--environment",
        type=str,
        help="Deployment environment. When specified, values from <experiment_name>.<environment>.yaml "
        "are used to override values from <experiment_name>.yaml.",
        default=None,
    )
    parser.add_argument(
        "--force-az-cli-credentials",
        help="Force the usage of the Az CLI credentials. Default value is False,"
        "in which case DefaultAzureCredential() is used.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--evaluators",
        type=str,
        help="Defines the evalutors to run. (* for all or comma separated list)",
        default="*",
    )
    parser.add_argument(
        "--identity",
        help="Set the identity type to use for the run (managed or user_identity). See https://microsoft.github.io/promptflow/reference/run-yaml-schema-reference.html",
        required=False,
        default=None,
        type=str,
    )

    parser.add_argument(
        "--client-id",
        help="Set the client id for managed identity. See https://microsoft.github.io/promptflow/reference/run-yaml-schema-reference.html",
        required=False,
        default=None,
        type=str,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Resolve the base path of the experiment
    exp_base_path = resolve_exp_base_path(args.base_path)

    run_identity = RunIdentityConfig(args.identity, args.client_id)

    # Call the module function with parsed arguments
    run_evaluation_flow(
        args.run_id,
        EvaluatorSelector.from_args(args.evaluators),
        run_identity,
        args.file,
        exp_base_path,
        args.report_dir,
        args.output_file,
        args.tags,
        args.subscription_id,
        args.environment,
        args.force_az_cli_credentials,
    )


if __name__ == "__main__":
    # Load variables from .env file into the environment
    load_dotenv(override=True)

    main()
