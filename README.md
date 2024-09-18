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
   export PYTHONPATH="."
   python llmops/src/run_standard_flow.py --file hello_world_experiment.yaml
   python llmops/src/run_standard_flow.py --file answer_correctness_experiment.yaml --evaluate --output-file run_id.txt
   ```


## Getting the dataset

We can use the [TruthfulQA dataset](https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv)

1. Download the CSV dataset from this [link](https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv)
2. Transform the dataset into jsonl and keep the first 50 lines
   ```
   python .\helpers\csv_to_jsonl.py ..\path\to\your\TruthfulQA.csv truthfulqa.jsonl
   ```

## Running Answer Correctness flow

To run the [Answer Correctness](https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html) flow on the [Amnesty QA data](https://huggingface.co/datasets/explodinggradients/amnesty_qa), run the following, from within the virtual environment:

>Note: This requires both a GPT-3 **and** an embeddings model deployed.
```
python .\llmops\src\run_standard_flow.py --file .\answer_corectness_amnesty_qa.yaml
```
