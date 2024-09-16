# The experiment.yaml file

The `experiment.yaml` is used to configure an LLM use-case. At least one `experiment.yaml` file is needed per use-case and it configures:
- The use-case (experiment) name (`name` block)
- The path to the "standard" use-case flow (`flow` block)
- The datasets used to test the flow (`datasets` block)
- The path(s) to the "evaluation" flows of the use-case (`evaluators` block)
- The datasets used to evaluate the flow (also `evaluators` block)

The full specification of the `experiment.yaml` file can be found [here](./experiment.yaml).
