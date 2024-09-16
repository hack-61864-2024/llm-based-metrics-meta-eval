import os
from typing import Any, List, Optional, Tuple

import yaml

_FLOW_DAG_FILENAME = "flow.dag.yaml"
_DEFAULT_FLOWS_DIR = "flows"


class Dataset:
    """
    Defines an evaluation dataset
    Can point to local file or an Azure ML Dataset.

    :param name: Unique name given to the dataset.
    :type name: str
    :param source: File path or Azure ML name and version (example: azureml:<dataset_name_in_azureml>:<version>).
    :type source: str
    """

    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source

    def with_mappings(self, mappings: dict[str, str]) -> "MappedDataset":
        return MappedDataset(mappings, self)

    def get_friendly_name(self) -> str:
        if self.name:
            return self.name

        data_ref = self.source.replace("azureml:", "")
        return data_ref.split(":", 1)[1]

    # Define equality operation
    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return NotImplemented

        return self.name == other.name and self.source == other.source


class MappedDataset:
    """
    Defines an Azure ML Mapped dataset referencing a Dataset object and a dictionary mapping prompt flow inputs to
    dataset columns or previous run outputs.

    :param dataset: Dataset to be mapped.
    :type dataset: Dataset
    :param mappings: Dictionary mapping prompt flow inputs to dataset columns or previous run outputs
    (example: {"col_0": "${data:col_0}", "col_1": "${run.outputs.col_1}"}).
    :type mappings: dict[str, str]
    """

    def __init__(self, mappings: dict[str, str], dataset: Dataset):
        self.dataset = dataset
        self.mappings = mappings

    # Define equality operation
    def __eq__(self, other):
        if not isinstance(other, MappedDataset):
            return NotImplemented

        return self.dataset == other.dataset and self.mappings == other.mappings


class Evaluator:
    """
    Defines a prompt flow evaluator flow.

    :param name: Name of the evaluation flow.
    :type name: str
    :param path: Path to the evaluation flow. Default value is "flows".
    :type path: str
    :param datasets: List of mapped datasets used for the evaluator flow.
    :type datasets: List[MappedDataset]
    """

    def __init__(self, name: str, datasets: list[MappedDataset], path: Optional[str] = None):
        self.name = name
        self.path = path or os.path.join("flows", name)
        self.datasets = datasets

    def find_dataset_mapping(self, dataset_name: str) -> Optional[MappedDataset]:
        for mapped_dataset in self.datasets:
            if mapped_dataset.dataset.source == dataset_name:
                return mapped_dataset

            # Workaround to compare local file with Azure ML dataset
            # Comparison must be improved to ensure that not only file/directory content as same files, but also same content
            if os.path.basename(mapped_dataset.dataset.source) == os.path.basename(dataset_name):
                return mapped_dataset

        return None

    # Define equality operation
    def __eq__(self, other):
        if not isinstance(other, Evaluator):
            return NotImplemented

        return self.name == other.name and self.datasets == other.datasets and self.path == other.path


class Metric:
    """
    Defines an evaluation metric.

    :param name: Name of the metric.
    :type name: str
    :param min_value: Minimum acceptable value of the metric.
    :type min_value: float
    """

    def __init__(self, name: str, min_value: float):
        self.name = name
        self.min_value = min_value

    def __eq__(self, other):
        if not isinstance(other, Metric):
            return NotImplemented

        return self.name == other.name and self.min_value == other.min_value


class RuntimeEnvironment:
    """
    Defines the Azure ML environment for an experiment.
    """

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = str(version)

    @classmethod
    def from_dict(cls, raw_environment: dict) -> Optional["RuntimeEnvironment"]:
        if raw_environment:
            name = raw_environment.get("name", None)
            version = raw_environment.get("version", None)
            if name and version:
                return cls(name=name, version=version)

        return None


class FlowDetail:
    """
    Contains the details of a flow (location, nodes, variants)

    :param flow_path: Path to a prompt flow flow.
    :type flow_path: str
    :param all_variants: List of dictionaries, one per llm node, from llm node variant name to llm node name.
    :type all_variants: list[dict[str, Any]]
    :param all_llm_nodes: Set of llm node names.
    :type all_llm_nodes: set
    :param default_variants: Dictionary from llm node name to default node variant.
    :type default_variants: dict[str, str]
    """

    def __init__(
        self,
        flow_path: str,
        all_variants: list[dict[str, Any]],
        all_llm_nodes: set,
        default_variants: dict,
    ):
        self.flow_path = flow_path
        self.all_variants = all_variants
        self.all_llm_nodes = all_llm_nodes
        self.default_variants = default_variants


class Experiment:
    """
    Contains the details of an experiment (name, flow, path, datasets, evaluators)

    :param base_path: Path to the directory containing the yaml file defining the experiment. Default value is current
    working directory.
    :type base_path: Optional[str]
    :param name: Name of the experiment.
    :type name: str
    :param flow: Name of the flow. Default value is :param name:.
    :type flow: Optional[str]
    :param datasets: List of datasets.
    :type datasets: List[Dataset]
    :param evaluators: List of evaluators.
    :type evaluators: List[Evaluator]
    """

    def __init__(
        self,
        base_path: Optional[str],
        name: str,
        flow: Optional[str],
        datasets: list[MappedDataset],
        evaluators: list[Evaluator],
        metrics: list[Metric],
        environment: Optional[RuntimeEnvironment] = None,
    ):
        self.base_path = base_path
        self.name = name
        self.flow = flow or name
        self.datasets = datasets
        self.evaluators = evaluators
        self.metrics = metrics
        self.environment = environment
        self._flow_detail: Optional[FlowDetail] = None

    def get_flow_detail(self) -> FlowDetail:
        self._flow_detail = self._flow_detail or self._load_flow_detail()
        return self._flow_detail

    def _load_flow_detail(self) -> FlowDetail:
        """
        Load flow details from the yaml files describing the experiment and the flow.
        """
        # Load flow data
        flow_path = _resolve_flow_dir(self.base_path, self.flow)
        flow_file_path = os.path.join(flow_path, _FLOW_DAG_FILENAME)
        yaml_data: dict
        with open(flow_file_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Find prompt variants and nodes
        all_variants: list[dict[str, Any]] = []
        all_llm_nodes = set()
        default_variants = {}
        for node_name, node_data in yaml_data.get("node_variants", {}).items():
            node_variant_mapping = {}
            variants = node_data.get("variants", {})
            default_variant: str = node_data["default_variant_id"]
            default_variants[node_name] = default_variant
            for variant_name, _ in variants.items():
                node_variant_mapping[variant_name] = node_name
                all_llm_nodes.add(node_name)
            all_variants.append(node_variant_mapping)

        for nodes in yaml_data["nodes"]:
            node_variant_mapping = {}
            if nodes.get("type", {}) == "llm":
                all_llm_nodes.add(nodes["name"])

        return FlowDetail(flow_path, all_variants, all_llm_nodes, default_variants)


# Helper function for raising errors
def _raise_error_if_missing_keys(keys: List[str], config: dict, message: str):
    for key in keys:
        if key not in config:
            raise ValueError(f"{message}: {key}")


def _create_datasets_and_default_mappings(
    raw: list[dict],
) -> Tuple[dict[str, Dataset], list[MappedDataset]]:
    """
    Create datasets and mapped datasets from list of dictionaries.

    :param raw: List of dictionaries containing the description of the experiment datasets.
    :type raw: list[dict]
    :return: Tuple of dictionary from dataset name to dataset and list of mapped datasets
    :rtype: Tuple[dict[str, Dataset], list[MappedDataset]]
    """
    datasets: dict[str, Dataset] = {}
    mappings: list[MappedDataset] = []
    for ds in raw:
        # Raise error if expected dataset configuration missing
        _raise_error_if_missing_keys(
            ["name", "source", "mappings"],
            ds,
            message="Dataset config missing parameter",
        )
        dataset = Dataset(ds["name"], ds["source"])
        datasets[dataset.name] = dataset
        mappings.append(dataset.with_mappings(ds["mappings"]))

    return datasets, mappings


def _create_evaluators(
    raw_evaluators: list[dict], datasets: dict[str, Dataset], base_path: Optional[str]
) -> list[Evaluator]:
    """
    Create evaluators from a list of dictionaries describing the raw evaluators, a dictionary of existing datasets,
    and the path to the evaluator flow.

    :param raw_evaluators: List of dictionaries containing the description of the experiment evaluators.
    :type raw_evaluators: list[dict]
    :param datasets: Dictionary from dataset name to Dataset object.
    :type datasets: dict[str, Dataset]
    :param base_path: Path to the evaluator flow directory containing the yaml description.
    Default value is current working directory.
    :type base_path: Optional[str]
    :return: List of evaluators.
    :rtype: list[Evaluator]
    """
    evaluators: list[Evaluator] = []
    for raw_evaluator in raw_evaluators:
        # Raise error if expected evaluator configuration missing
        _raise_error_if_missing_keys(
            ["name", "datasets"],
            raw_evaluator,
            message="Evaluator config missing parameter",
        )
        evaluator_datasets: list[MappedDataset] = []
        eval_name = raw_evaluator["name"]
        for ds in raw_evaluator["datasets"]:
            # Raise error if expected evaluator dataset configuration missing
            _raise_error_if_missing_keys(["mappings"], ds, message="Evaluator dataset config missing parameter")
            dataset_name = ds["name"]
            # Raise error if expected evaluator dataset not found in the list of experiment datasets
            dataset = datasets.get(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_name} not found in evaluator {eval_name}")
            # Use experiment dataset with evaluator mappings
            evaluator_datasets.append(dataset.with_mappings(ds["mappings"]))
        flow = raw_evaluator.get("flow") or eval_name
        flow_path = _resolve_flow_dir(base_path, flow)
        evaluator = Evaluator(name=eval_name, datasets=evaluator_datasets, path=flow_path)
        evaluators.append(evaluator)
    return evaluators


def _create_metrics(raw_metrics: list[dict]) -> list[Metric]:
    """
    Create metrics from a list of dictionaries describing the metric name and minimum acceptable value.

    :param raw_metrics: List of dictionaries containing the description of the experiment metrics.
    :type raw_evaluators: list[dict]
    :return: List of metrics.
    :rtype: list[Metric]
    """
    metrics: list[Metric] = []
    for raw_metric in raw_metrics:
        _raise_error_if_missing_keys(
            ["name", "minimum_value"],
            raw_metric,
            message="Metrics config missing parameter",
        )
        metrics.append(Metric(raw_metric["name"], raw_metric["minimum_value"]))
    return metrics


def _resolve_flow_dir(base_path: Optional[str], flow: str) -> str:
    """
    Resolve path to the yaml file describing the flow. Tries multiple resolution methods.
    - If base_path not provided, uses the current working directory
    - Looks for "flow.dag.yaml" in the base_path/flow, returns that path if found; otherwise returns
      base_path/flows/flow.

    :param base_path: Path to the flow directory. Default value is current working directory.
    :type base_path: Optional[str]
    :param flow: Name of the flow.
    :type flow: str
    :return: Path to flow.
    :rtype: str
    """
    # Check if the flow path can be deducted from base path
    safe_base_path = base_path or ""
    if os.path.isfile(os.path.join(safe_base_path, flow, _FLOW_DAG_FILENAME)):
        return os.path.abspath(os.path.join(safe_base_path, flow))

    return os.path.join(safe_base_path, _DEFAULT_FLOWS_DIR, flow)


def load_experiment(
    filename: Optional[str] = None,
    base_path: Optional[str] = None,
    env: Optional[str] = None,
) -> Experiment:
    """
    Load an experiment from a YAML file.

    :param filename: The experiment filename. Default is "experiment.yaml"
    :type filename: Optional[str]
    :param base_path: The base path of the experiment. If not specified, the current working directory is used.
    :type base_path: Optional[str]
    :param env: The environment to use. If specified, look for an overlay experiment file named
    <experiment_name>.<env>.yaml and use it to override experiment dataset sources.
    :type env: Optional[str]
    """

    experiment_file_name = filename or "experiment.yaml"
    safe_base_path = base_path or ""

    exp_file_path = os.path.join(safe_base_path, experiment_file_name)
    exp_config: dict
    with open(exp_file_path, "r") as yaml_file:
        exp_config = yaml.safe_load(yaml_file)

    # Read raw datasets
    raw_datasets: list[dict] = exp_config["datasets"]
    if not raw_datasets:
        raise ValueError("No datasets configured for experiment")

    # Read raw environment
    raw_environment: dict = exp_config.get("environment", {})

    # Override from environment specific overlay file <experiment_name>.<env>.yaml
    if env:
        _apply_env_overlays(env, experiment_file_name, safe_base_path, raw_datasets, raw_environment)

    # Create datasets and mappings
    datasets, mappings = _create_datasets_and_default_mappings(raw_datasets)

    # Read raw evaluators and create evaluator list
    raw_evaluators: list[dict] = exp_config.get("evaluators", [])
    evaluators: list[Evaluator] = []
    if raw_evaluators is not None and len(raw_evaluators) > 0:
        evaluators = _create_evaluators(raw_evaluators, datasets, base_path)

    # Read raw metrics and create metric list
    raw_metrics: list[dict] = exp_config.get("metrics", [])
    metrics = _create_metrics(raw_metrics)

    environment = RuntimeEnvironment.from_dict(raw_environment) if raw_environment else None

    # Create experiment
    return Experiment(
        base_path=base_path,
        name=exp_config["name"],
        flow=exp_config.get("flow"),
        datasets=mappings,
        evaluators=evaluators,
        metrics=metrics,
        environment=environment,
    )


def _apply_env_overlays(  # noqa: C901
    env: str,
    experiment_file_name: str,
    safe_bath_path: str,
    raw_datasets: list[dict],
    raw_environment: dict,
):
    """
    Apply environment overlays if a file with the name <experiment_name>.<env>.yaml exists.

    :param experiment_file_name: The base experiment filename.
    :type experiment_file_name: str
    :param safe_base_path: The path to the directory containing the environment specific experiment overlay yaml file.
    :type safe_base_path: str
    :param raw_datasets: The raw description of the datasets from the base experiment yaml file.
    :type raw_datasets: list[dict]
    :param raw_environment: The raw runtime environment from the base experiment yaml file.
    :type raw_environment: dict
    :raises ValueError: If attempted override variable is not supported.
    """
    file_parts = os.path.splitext(experiment_file_name)
    if len(file_parts) != 2:  # noqa: PLR2004
        raise ValueError(f"Invalid experiment filename '{experiment_file_name}'")

    env_exp_file_path = os.path.join(safe_bath_path, f"{file_parts[0]}.{env}{file_parts[1]}")
    if not os.path.exists(env_exp_file_path):
        return

    env_overrides: dict
    with open(env_exp_file_path, "r") as yaml_file:
        env_overrides = yaml.safe_load(yaml_file)

    if "datasets" in env_overrides:
        env_datasets = env_overrides["datasets"]
        for env_ds in env_datasets:
            if "mapping" in env_ds:
                raise ValueError("Can not override dataset mappings")

            # If the dataset is available in the raw datasets, override the source
            for raw_ds in raw_datasets:
                if raw_ds["name"] == env_ds["name"]:
                    raw_ds["source"] = env_ds["source"]
                    break

    if "evaluators" in env_overrides:
        raise ValueError("Can not override evaluators")

    if "metrics" in env_overrides:
        raise ValueError("Can not override metrics")

    if "environment" in env_overrides:
        raw_environment.update(env_overrides["environment"])
