# llm-based-metrics-meta-eval
The goal of this repository is to evaluate the performance of some LLMs for the task generating report for evaluation.

## Tools 
- https://github.com/explodinggradients/ragas

## Getting started

1. Create and activate the python virtual environment
   ```
   python -m venv .venv
   ./.venv/Scripts/activate
   ```

2. Install the requirements
   ```
   python -m pip install -r requirements.txt
   ```

3. Install the VSCode promptflow extension

4. Add the root directory to pythonpath 
   ```
   $env:PYTHONPATH="."
   ```

5. Copy the `.env.example` file into  a new `.env` and fill in your variables.

6. Login with Azure CLI
   ```
   az login
   ```

7. Run the hello-world flow
   ```
   python .\llmops\src\run_standard_flow.py --file .\hello_world_experiment.yaml
   ```


## Getting the dataset

We can use the [TruthfulQA dataset](https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv)

1. Download the CSV dataset from this [link](https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv)
2. Transform the dataset into jsonl and keep the first 50 lines
   ```
   python .\helpers\csv_to_jsonl.py ..\path\to\your\TruthfulQA.csv truthfulqa.jsonl
   ```
