import os
from pathlib import Path
from typing import Any, List

import pytest

from llmops.src.experiment import (
    Dataset,
    Evaluator,
    Experiment,
    MappedDataset,
    Metric,
    _create_datasets_and_default_mappings,
    _create_evaluators,
    _create_metrics,
    load_experiment,
)

THIS_PATH = Path(__file__).parent
RESOURCE_PATH = THIS_PATH / "resources"


def check_lists_equal(actual: List[Any], expected: List[Any]):
    assert len(actual) == len(expected)
    assert all(any(a == e for a in actual) for e in expected)
    assert all(any(a == e for e in expected) for a in actual)


def test_create_datasets_and_default_mappings():
    # Prepare inputs
    g_name = "groundedness"
    g_source = "groundedness_source"
    g_mappings = {"claim": "claim_mapping"}
    g_dataset = Dataset(g_name, g_source)

    r_name = "recall"
    r_source = "recall_source"
    r_mappings = {"input": "input_mapping", "gt": "gt_mapping"}
    r_dataset = Dataset(r_name, r_source)

    raw_datasets = [
        {"name": g_name, "source": g_source, "mappings": g_mappings},
        {"name": r_name, "source": r_source, "mappings": r_mappings},
    ]

    # Prepare expected outputs
    expected_datasets = {g_name: g_dataset, r_name: r_dataset}
    expected_mapped_datasets = [MappedDataset(g_mappings, g_dataset), MappedDataset(r_mappings, r_dataset)]

    # Check outputs
    [datasets, mapped_datasets] = _create_datasets_and_default_mappings(raw_datasets)
    assert datasets == expected_datasets
    check_lists_equal(mapped_datasets, expected_mapped_datasets)


@pytest.mark.parametrize(
    "raw_datasets",
    [
        [{}],
        [
            {
                "name": "groundedness",
            }
        ],
        [
            {
                "name": "groundedness",
                "source": "groundedness_source",
            }
        ],
    ],
)
def test_create_datasets_and_default_mappings_missing_parameters(raw_datasets: List[dict]):
    # Check that datasets with missing parameters raise an exception
    with pytest.raises(ValueError, match="Dataset config missing parameter"):
        _create_datasets_and_default_mappings(raw_datasets)


def test_create_evaluators():
    # Prepare inputs
    g_name = "groundedness"
    g_flow = "groundedness_eval"
    g_source = "groundedness_source"
    g_mappings = {"claim": "claim_mapping"}
    g_dataset = Dataset(g_name, g_source)

    r_name = "recall"
    r_source = "recall_source"
    r_mappings = {"input": "input_mapping", "gt": "gt_mapping"}
    r_dataset = Dataset(r_name, r_source)

    available_datasets = {g_name: g_dataset, r_name: r_dataset}

    raw_evaluators = [
        {"name": g_name, "flow": g_flow, "datasets": [{"name": g_dataset.name, "mappings": g_mappings}]},
        {"name": r_name, "datasets": [{"name": r_dataset.name, "mappings": r_mappings}]},
    ]

    # Test with base_path
    base_path = "/path/to/flow/"

    # Prepare expected outputs
    expected_evaluators = [
        Evaluator(g_name, [MappedDataset(g_mappings, g_dataset)], os.path.join(base_path, "flows", g_flow)),
        Evaluator(r_name, [MappedDataset(r_mappings, r_dataset)], os.path.join(base_path, "flows", r_name)),
    ]

    # Check outputs
    evaluators = _create_evaluators(raw_evaluators, available_datasets, base_path)
    assert evaluators == expected_evaluators
    assert evaluators[0].find_dataset_mapping(g_source) == MappedDataset(g_mappings, g_dataset)
    assert evaluators[0].find_dataset_mapping(r_source) is None
    assert evaluators[1].find_dataset_mapping(r_source) == MappedDataset(r_mappings, r_dataset)
    assert evaluators[1].find_dataset_mapping(g_source) is None

    # Test without base_path
    base_path = None

    # Prepare expected outputs
    expected_evaluators = [
        Evaluator(g_name, [MappedDataset(g_mappings, g_dataset)], os.path.join("flows", g_flow)),
        Evaluator(r_name, [MappedDataset(r_mappings, r_dataset)], os.path.join("flows", r_name)),
    ]

    # Check outputs
    evaluators = _create_evaluators(raw_evaluators, available_datasets, base_path)
    assert evaluators == expected_evaluators


@pytest.mark.parametrize(
    "raw_evaluators",
    [
        [{}],
        [
            {
                "name": "groundedness",
            }
        ],
        [
            {
                "name": "groundedness",
                "datasets": [{"name": "groundedness"}],
            }
        ],
    ],
)
def test_create_evaluators_missing_parameters(raw_evaluators: List[dict]):
    available_datasets = {
        "groundedness": Dataset("groundedness", "groundedness_source"),
    }
    # Check that evaluators with missing parameters raise an exception
    with pytest.raises(ValueError, match=".*missing parameter"):
        _create_evaluators(raw_evaluators, available_datasets, None)


def test_create_evaluators_invalid_dataset():
    # Prepare inputs
    eval_name = "groundedness"
    dataset_name = "groundedness"

    raw_evaluators = [
        {
            "name": eval_name,
            "flow": "groundedness_eval",
            "datasets": [{"name": dataset_name, "mappings": {"claim": "claim_mapping"}}],
        }
    ]

    available_datasets = {}

    # Check that evaluators with invalid datasets (datasets not in the available_datasets dict) raise an exception
    with pytest.raises(ValueError, match=f"Dataset {dataset_name} not found in evaluator {eval_name}"):
        _create_evaluators(raw_evaluators, available_datasets, None)


def test_create_metrics():
    # Prepare inputs
    m1_name = "metric1"
    m1_value = 0.76
    m2_name = "metric2"
    m2_value = 0.53

    raw_metrics = [
        {
            "name": m1_name,
            "minimum_value": m1_value,
        },
        {
            "name": m2_name,
            "minimum_value": m2_value,
        },
    ]

    # Prepare expected outputs
    expected_metrics = [
        Metric(m1_name, m1_value),
        Metric(m2_name, m2_value),
    ]

    # Check outputs
    metrics = _create_metrics(raw_metrics)
    check_lists_equal(metrics, expected_metrics)


def test_experiment_creation():
    # Prepare inputs
    base_path = str(RESOURCE_PATH)
    name = "exp_name"
    flow = "exp_flow"

    # Prepare expected outputs
    expected_flow_variants = [
        {"var_0": "node_var_0", "var_1": "node_var_0"},
        {"var_3": "node_var_1", "var_4": "node_var_1"},
    ]
    expected_flow_default_variants = {"node_var_0": "var_0", "node_var_1": "var_3"}
    expected_flow_llm_nodes = {
        "node_var_0",
        "node_var_1",
    }

    # Check outputs
    experiment = Experiment(base_path, name, flow, [], [], [])
    flow_detail = experiment.get_flow_detail()

    assert flow_detail.flow_path == os.path.join(base_path, "flows", flow)
    assert flow_detail.all_variants == expected_flow_variants
    assert flow_detail.default_variants == expected_flow_default_variants
    assert flow_detail.all_llm_nodes == expected_flow_llm_nodes


def test_load_experiment():
    # Prepare inputs
    base_path = str(RESOURCE_PATH)

    # Prepare expected outputs
    expected_name = "exp"
    expected_flow = "exp_flow"

    expected_dataset_names = ["ds1", "ds2"]
    expected_dataset_sources = ["ds1_source", "ds2_source"]
    expected_dataset_mappings = [{"ds1_input": "ds1_mapping"}, {"ds2_input": "ds2_mapping"}]
    expected_datasets = [
        Dataset(expected_dataset_names[0], expected_dataset_sources[0]),
        Dataset(expected_dataset_names[1], expected_dataset_sources[1]),
    ]
    expected_mapped_datasets = [
        MappedDataset(expected_dataset_mappings[0], expected_datasets[0]),
        MappedDataset(expected_dataset_mappings[1], expected_datasets[1]),
    ]

    expected_evaluator_dataset_mappings = [
        {"ds1_input": "ds1_mapping", "ds1_extra": "ds1_extra_mapping"},
        {"ds2_extra": "ds2_extra_mapping"},
        {"ds2_input": "ds2_diff_mapping"},
    ]
    expected_evaluator_mapped_datasets = [
        [
            MappedDataset(expected_evaluator_dataset_mappings[0], expected_datasets[0]),
            MappedDataset(expected_evaluator_dataset_mappings[1], expected_datasets[1]),
        ],
        [MappedDataset(expected_evaluator_dataset_mappings[2], expected_datasets[1])],
    ]
    expected_evaluators = [
        Evaluator("eval1", expected_evaluator_mapped_datasets[0], os.path.join(base_path, "flows", "eval1")),
        Evaluator("eval2", expected_evaluator_mapped_datasets[1], os.path.join(base_path, "flows", "eval2")),
    ]

    expected_metrics = [Metric("metric1", 0.76), Metric("metric2", 0.53)]

    # Test with no environment overrides
    # Check outputs
    experiment = load_experiment(base_path=base_path)
    assert experiment.base_path == base_path
    assert experiment.name == expected_name
    assert experiment.flow == expected_flow
    assert experiment.datasets == expected_mapped_datasets
    assert experiment.evaluators == expected_evaluators
    assert experiment.metrics == expected_metrics

    # Test with environment overrides
    # Modify expected outputs
    expected_overridden_dataset_source = "overridden_ds1_source"
    expected_overridden_dataset = Dataset(expected_dataset_names[0], expected_overridden_dataset_source)
    expected_overridden_mapped_datasets = [
        MappedDataset(expected_dataset_mappings[0], expected_overridden_dataset),
        MappedDataset(expected_dataset_mappings[1], expected_datasets[1]),
    ]
    expected_overridden_evaluator_mapped_datasets = [
        MappedDataset(expected_evaluator_dataset_mappings[0], expected_overridden_dataset),
        MappedDataset(expected_evaluator_dataset_mappings[1], expected_datasets[1]),
    ]
    expected_overridden_evaluators = [
        Evaluator("eval1", expected_overridden_evaluator_mapped_datasets, os.path.join(base_path, "flows", "eval1")),
        Evaluator("eval2", expected_evaluator_mapped_datasets[1], os.path.join(base_path, "flows", "eval2")),
    ]

    # Check outputs
    experiment = load_experiment(base_path=base_path, env="dev")
    assert experiment.base_path == base_path
    assert experiment.name == expected_name
    assert experiment.flow == expected_flow
    assert experiment.datasets == expected_overridden_mapped_datasets
    assert experiment.evaluators == expected_overridden_evaluators
    assert experiment.metrics == expected_metrics

    ## Test experiment with no evaluators
    experiment = load_experiment(filename="experiment_no_eval.yaml", base_path=base_path)
    assert experiment.base_path == base_path
    assert experiment.name == expected_name
    assert experiment.flow == expected_flow
    assert experiment.datasets == expected_mapped_datasets
    assert experiment.evaluators == []
    assert experiment.metrics == []
