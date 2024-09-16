import argparse
import ast
import datetime
import os
from enum import Enum
from typing import Optional

import pandas as pd
from dotenv import dotenv_values, load_dotenv
from promptflow.azure import PFClient
from promptflow.entities import Run

from llmops.src.common import (
    RUN_TAG_ENVIRONMENT,
    GitRepositoryStatus,
    RunIdentityConfig,
    RunName,
    create_run_tags,
    generate_random_name_part,
    get_credentials,
    get_logger,
    resolve_exp_base_path,
    wait_job_finish,
)
from llmops.src.experiment import load_experiment
from llmops.src.experiment_cloud_config import ExperimentCloudConfig

logger = get_logger()


def check_dictionary_contained(ref_dict, dict_list):
    for candidate_dict in dict_list:
        set1 = {frozenset(dict(candidate_dict).items())}
        set2 = {frozenset(ref_dict.items())}
        if set1 == set2:
            return True
    return False


class VariantsSelector:
    """
    Selects the variants to run. Options are default, all or custom.
    """

    class VariantSelectionOption(Enum):
        DEFAULTS_ONLY = 1
        ALL = 2
        CUSTOM = 3

    def __init__(
        self,
        selector: VariantSelectionOption,
        selected_variants: Optional[list[str]] = None,
    ):
        self._selector = selector
        self._selected_variants = selected_variants or []

    @property
    def defaults_only(self) -> bool:
        return self._selector == self.VariantSelectionOption.DEFAULTS_ONLY

    def is_variant_enabled(self, node: str, variant: str) -> bool:
        if self._selector in [
            VariantsSelector.VariantSelectionOption.DEFAULTS_ONLY,
            VariantsSelector.VariantSelectionOption.ALL,
        ]:
            return True

        for selected_variant in self._selected_variants:
            if selected_variant in (variant, f"{node}.{variant}"):
                return True
        return False

    @classmethod
    def from_args(cls, variants: str):
        variants = variants.strip().lower()
        if variants in ["*", "all"]:
            return cls(cls.VariantSelectionOption.ALL)
        if variants in ["defaults", "default"]:
            return cls(cls.VariantSelectionOption.DEFAULTS_ONLY)
        return cls(cls.VariantSelectionOption.CUSTOM, [v.strip() for v in variants.split(",")])


class DatasetSelector:
    """
    Selects the datasets to run. Options are default, all or custom.
    """

    def __init__(self, datasets: Optional[list[str]]) -> None:
        self._enabled_datasets = datasets

    def is_dataset_enabled(self, dataset: str) -> bool:
        return self._enabled_datasets is None or dataset.lower() in self._enabled_datasets

    @classmethod
    def from_args(cls, datasets: str):
        datasets = datasets.strip().lower()
        if datasets is None or datasets in ["*", "all"]:
            return cls(None)

        return cls([d.strip().lower() for d in datasets.split(",")])


def run_flow(  # noqa C901
    variants_selector: VariantsSelector,
    datasets_selector: DatasetSelector,
    run_identity: RunIdentityConfig,
    exp_filename: Optional[str] = None,
    exp_base_path: Optional[str] = None,
    output_file: Optional[str] = None,
    report_dir: Optional[str] = None,
    tags: Optional[str] = None,
    subscription_id: Optional[str] = None,
    environment: Optional[str] = None,
    suffix: Optional[str] = None,
    force_az_cli_credentials: bool = False,
):
    """
    Run standard flow of experiment for all selected variants with all selected datasets.

    In the event that a flow has multiple LLM nodes, with multiple variants each, then every selected variant
    from every node will run ONLY with the default variant of all other nodes. So NOT all possible combinations
    will be executed.

    Example: A flow has 2 LLM nodes a and b and each node has 2 variants 0 and 1 with 0 being the default variant.
    We select all variants from each node. The following combinations will run [{a0, b0}, {a1, b0}, {a0, b1}].
    The combination {a1, b1} will be skipped, because neither of the 2 variants are the default variant for the
    corresponding node.

    If we also take into account 2 selected dataset d0 and d1, the following run combinations will be tested
    [{a0, b0, d0}, {a1, b0, d0}, {a0, b1, d0}, {a0, b0, d1}, {a1, b0, d1}, {a0, b1, d1}]

    WORKSPACE_NAME, RESOURCE_GROUP_NAME expected as environment variables. SUBSCRIPTION_ID expected as environment
    variable or argument.

    :param variants_selector: Selection of node variants to run.
    :type variants_selector: VariantsSelector
    :param datasets_selector: Selection of dataset variants to run.
    :type datasets_selector: DatasetSelector
    :param exp_filename: Name of the yaml file defining the experiment. Defaults to None, in which case
    "experiment.yaml" is used.
    :type exp_filename: Optional[str]
    :param exp_base_path: Path to the yaml file defining the experiment. Defaults to None, in which case
    current working directory is used.
    :type exp_base_path: Optional[str]
    :param output_file: Path to file to write the version of the run IDs of the standard flows.
    Defaults to None, in which case the run IDs are not saved.
    :type output_file: Optional[str]
    :param tags: String reprisenting the dictionary of tags to be applied to the standard run.
    :type tags: Optional[str]
    :param subscription_id: Subscription ID, overwrites the SUBSCRIPTION_ID environment variable.
    Defaults to None.
    :type subscription_id: Optional[str]
    :param environment: Used environment ('dev', 'prd' etc.). Defaults to None.
    :type environment: Optional[str]
    :param suffix: Run name suffix. Defaults to None.
    :type suffix: Optional[str]
    :param force_az_cli_credentials: Force the usage of the Az CLI credentials. Default value is False,
    in which case DefaultAzureCredential() is used.
    :type force_az_cli_credentials: bool
    """
    # Load all necessary variables
    if not environment:
        # In CI we want to load the content of the environment variable if environment is not set
        environment = "dev" if os.getenv("CI") else "local"
    logger.info("Loading experiment for %s environment", environment)
    experiment = load_experiment(filename=exp_filename, base_path=exp_base_path, env=environment)
    exp_filename = experiment.name

    config = ExperimentCloudConfig(subscription_id)
    common_tags = {} if not tags else ast.literal_eval(tags)
    common_tags[RUN_TAG_ENVIRONMENT] = environment

    # Initialize clients
    pf = PFClient(
        get_credentials(force_az_cli_credentials),
        config.subscription_id,
        config.resource_group_name,
        config.workspace_name,
    )

    # Load flow data
    flow_detail = experiment.get_flow_detail()
    run_ids = []
    past_runs = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name_suffixes: list[str] = []
    if suffix:
        name_suffixes.append(suffix.strip())
    name_suffixes.append(generate_random_name_part())
    git_status = GitRepositoryStatus.from_git()

    logger.info("Running experiment %s", experiment.name)
    run_results = []

    for mapped_dataset in experiment.datasets:
        # Get dataset name
        dataset = mapped_dataset.dataset
        if not datasets_selector.is_dataset_enabled(dataset.name):
            continue

        logger.info("Using dataset %s", mapped_dataset.dataset.source)

        column_mapping = mapped_dataset.mappings
        data_ref = dataset.get_friendly_name()
        env_vars = dotenv_values()
        env_vars["PF_LOGGING_LEVEL"] = "WARNING"
        identity = run_identity.to_identity()
        logger.info("Using identity %s", identity)

        # If we're using non-default variants, we need to select the varint when creating the run
        # otherise we can skip the selection and the default will be used automatically
        if len(flow_detail.all_variants) != 0 and not variants_selector.defaults_only:
            logger.info("Start processing %d variants", len(flow_detail.all_variants))
            for variant in flow_detail.all_variants:
                for variant_id, node_id in variant.items():
                    logger.info("Creating run for node '%s' variant '%s'", node_id, variant_id)
                    if not variants_selector.is_variant_enabled(node_id, variant_id):
                        continue

                    variant_string = f"${{{node_id}.{variant_id}}}"
                    run_tags = create_run_tags(
                        common_tags=common_tags,
                        node_id=node_id,
                        variant_id=variant_id,
                        git_status=git_status,
                        name_suffixes=name_suffixes,
                        dataset_name=data_ref,
                    )

                    get_current_defaults = {
                        key: value
                        for key, value in flow_detail.default_variants.items()
                        if key != node_id or value != variant_id
                    }
                    get_current_defaults[node_id] = variant_id
                    get_current_defaults["dataset"] = data_ref

                    # This check validates that we are not running the same combination of variants more than once
                    if not check_dictionary_contained(get_current_defaults, past_runs):
                        past_runs.append(get_current_defaults)
                        name = RunName.for_standard_run(
                            has_multiple_datasets=len(experiment.datasets) > 1,
                            experiment_name=exp_filename,
                            variant_id=variant_id,
                            data_ref=dataset.get_friendly_name(),
                            timestamp=timestamp,
                            git_status=git_status,
                            suffixes=name_suffixes,
                        )

                        # Create run object
                        if not config.runtime:
                            logger.info("Using automatic runtime and serverless compute for the prompt flow run")
                        else:
                            logger.info(
                                "Using runtime '%s' for the prompt flow run",
                                config.runtime,
                            )

                        run = Run(
                            flow=flow_detail.flow_path,
                            data=dataset.source,
                            variant=variant_string,
                            name=name.name,
                            display_name=name.display_name,
                            environment_variables=env_vars,
                            column_mapping=column_mapping,
                            tags=run_tags,  # type: ignore
                            runtime=config.runtime,
                            resources=None if config.runtime else {"instance_type": config.serverless_instance_type},
                            identity=identity,
                        )
                        run._experiment_name = exp_filename

                        # Execute the run
                        logger.info(
                            "Starting prompt flow run '%s' in Azure ML. This can take a few minutes.",
                            run.name,
                        )
                        job = pf.runs.create_or_update(run, stream=True)
                        run_ids.append(job.name)
                        wait_job_finish(job)

                        # Save run results
                        df_result = pf.get_details(job, all_results=True)
                        df_result["dataset"] = dataset.source
                        df_result["run_id"] = run.name
                        df_result["variant_id"] = variant_id
                        run_results.append(df_result.copy())

                        logger.info("Run %s completed with status %s", job.name, job.status)
                        logger.info("Results:\n%s", df_result.head(10))

            logger.info("Finished processing all variants")
        else:
            logger.info("Start processing default variant")
            name = RunName.for_standard_run(
                experiment_name=exp_filename,
                has_multiple_datasets=len(experiment.datasets) > 1,
                data_ref=dataset.get_friendly_name(),
                git_status=git_status,
                suffixes=name_suffixes,
                timestamp=timestamp,
            )

            run_tags = create_run_tags(
                common_tags=common_tags,
                variants=flow_detail.default_variants,
                git_status=git_status,
                name_suffixes=name_suffixes,
                dataset_name=data_ref,
            )

            # Create run object
            if not config.runtime:
                logger.info("Using automatic runtime and serverless compute for the prompt flow run")
            else:
                logger.info("Using runtime '%s' for the prompt flow run", config.runtime)

            run = Run(
                flow=flow_detail.flow_path,
                data=dataset.source,
                name=name.name,
                display_name=name.display_name,
                environment_variables=env_vars,
                column_mapping=column_mapping,
                tags=run_tags,  # type: ignore
                runtime=config.runtime,
                resources=None if config.runtime else {"instance_type": config.serverless_instance_type},
                identity=identity,
            )
            run._experiment_name = exp_filename

            # Execute the run
            logger.info(
                "Starting prompt flow run '%s' in Azure ML. This can take a few minutes.",
                run.name,
            )
            job = pf.runs.create_or_update(run, stream=True)
            run_ids.append(job.name)
            wait_job_finish(job)
            df_result = pf.get_details(job)

            # Save run results
            df_result = pf.get_details(job, all_results=True)
            df_result["dataset"] = dataset.source
            df_result["run_id"] = job.name
            run_results.append(df_result.copy())

            logger.info("Run %s completed with status %s", job.name, job.status)
            logger.info("Results:\n%s", df_result.head(10))
            logger.info("Finished processing default variant\n")

    if report_dir:
        logger.info("Writing reports to '%s'", report_dir)
        timestamped_report_dir = os.path.join(report_dir, timestamp)
        os.makedirs(timestamped_report_dir, exist_ok=True)

        # Write the results of the evaluation flow (all datasets/standard runs) into csv and html files
        results_df = pd.concat(run_results, ignore_index=True)
        results_df["experiment_name"] = experiment.name

        results_df.to_csv(f"{timestamped_report_dir}/{experiment.name}_standard_flow_results.csv")
        with open(
            f"{timestamped_report_dir}/{experiment.name}_standard_flow_results.html",
            "w",
            encoding="utf-8",
        ) as results_html:
            results_html.write(results_df.to_html(index=False))

    logger.info("Completed runs: %s", str(run_ids))
    if output_file is not None:
        with open(output_file, "w") as out_file:
            out_file.write(str(run_ids))


def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser("run_standard_flow")
    parser.add_argument(
        "--file",
        type=str,
        help="The experiment file. Default is 'experiment.yaml'",
        required=False,
        default="experiment.yaml",
    )
    parser.add_argument(
        "--variants",
        type=str,
        help="Defines the variants to run. (* for all, defaults for all defaults, or comma separated list)",
        default="*",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Defines the datasets to use. (* for all, or comma separated list of dataset names)",
        default="*",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="A file to save run ids of the created Azure ML jobs",
        default=None,
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="A folder to save standard run results",
    )
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
        "--evaluate",
        action="store_true",
        help="Indicates if the evaluation should be run after the standard flow",
        default=False,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Indicates the name suffix for the run. Run name will be: "
        "[git branch] [optional: suffix] [random 6 characters]",
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

    # Run standard flow
    # Call the module function with parsed arguments
    run_flow(
        VariantsSelector.from_args(args.variants),
        DatasetSelector.from_args(args.datasets),
        run_identity,
        args.file,
        exp_base_path,
        args.output_file,
        args.report_dir,
        args.tags,
        args.subscription_id,
        args.environment,
        args.suffix,
        args.force_az_cli_credentials,
    )

    # Optionally evaluate the run immediately after
    if args.evaluate:
        logger.info("Starting automatic evaluation...")
        from llmops.src.run_evaluation_flow import EvaluatorSelector, run_evaluation_flow

        run_evaluation_flow(
            exp_filename=args.file,
            run_identity=run_identity,
            run_id=args.output_file,
            evaluator_selector=EvaluatorSelector.all(),
            report_dir=args.report_dir,
            output_file=None,
            tags=args.tags,
            subscription_id=args.subscription_id,
            exp_base_path=exp_base_path,
            environment=args.environment,
            force_az_cli_credentials=args.force_az_cli_credentials,
        )


if __name__ == "__main__":
    # Load variables from .env file into the environment
    load_dotenv(override=True)

    main()
