# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
from promptflow.core import tool


@tool
def line_process(groundtruth: str, prediction: str):
    """
    This tool processes the prediction of a single line and returns the processed result.

    :param groundtruth: the groundtruth of a single line.
    :param prediction: the prediction of a single line.
    """
    
    groundtruth_json = json.loads(groundtruth)
    print(groundtruth_json)
    prediction_json = json.loads(prediction)
    print(prediction_json)

    processed_results = []
    classifications = ["TP", "FP", "FN"]
    for classification in classifications:
        for gt_tp in groundtruth_json[classification]:
            found = False
            for p_tp in prediction_json[classification]:
                if gt_tp["answer"].lower() == p_tp["answer"].lower():
                    processed_results.append("Correct")
                    found = True
            if not found:
                processed_results.append("Incorrect")

        # Count extraneous classifications as Incorrect
        # for p_tp in prediction_json[classification]:
        #     found = False
        #     for gt_tp in groundtruth_json[classification]:
        #         if gt_tp["answer"].lower() == p_tp["answer"].lower():
        #             found = True
        #     if not found:
        #         processed_results.append("Incorrect")


    # Add your line processing logic here
    # processed_result = "Correct" if groundtruth.lower() == prediction.lower() else "Incorrect"

    return processed_results
