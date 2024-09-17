import pandas as pd

data_path = './english.json' # using data from https://huggingface.co/datasets/explodinggradients/amnesty_qa  english.json

def load_amnesty():
    """
    Loads the Amnesty QA dataset from a Parquet file and transforms it to a format expected by Regas.

    The function reads a Parquet file containing the Amnesty QA dataset, processes the data to 
    match the input format required by Regas, and returns a transformed DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'question', 'answer', 'contexts', and 'ground_truth'.

    Example:
        transformed_df = load_amnesty()
        print(transformed_df.head())
    """

    # Read the json file
    df = pd.read_json(data_path)
    return df
