# read csv file and convert it to jsonl file

import csv
import json
import sys


def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        with open(jsonl_file, "w") as f:
            # Only read the first 50 rows
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    csv_file = sys.argv[1]
    jsonl_file = sys.argv[2]
    csv_to_jsonl(csv_file, jsonl_file)
