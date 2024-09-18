# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import List

from promptflow.core import log_metric, tool


@tool
def aggregate(processed_results: List[float]) -> float:
    """
    This tool aggregates the processed result of all lines and calculate the average. Then logs the result to mlflow.

    :param processed_results: List of the output of correctness_evaluation node.
    """

    # Add your aggregation logic here
    # Aggregate the results of all lines and calculate the accuracy
    sum_correctness = sum(processed_results)
    aggregated_result = round((sum_correctness / len(processed_results)), 2)

    # Log metric the aggregate result
    log_metric(key="average_correctness", value=aggregated_result)

    return aggregated_result
