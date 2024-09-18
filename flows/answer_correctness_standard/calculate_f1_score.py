
import json
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(result: str) -> float:
    result_json = json.loads(result)
    tp_total = len(result_json["TP"])
    fp_total = len(result_json["FP"])
    fn_total = len(result_json["FN"])
    return tp_total / ( tp_total + ((fp_total + fn_total) * 0.5))
