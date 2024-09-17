import pandas as pd

parquetpath = './test-00000-of-00001.parquet' # using data from https://huggingface.co/datasets/MichaelR207/enron_qa_0822  test-00000-of-00001.parquet

def load_enron():
    """
    Loads the Enron dataset from a Parquet file and transforms it to a format expected by Regas.

    The function reads a Parquet file containing the Enron_QA dataset, processes the data to 
    match the input format required by Regas, and returns a transformed DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'question', 'answer', 'contexts', and 'ground_truth'.

    Example:
        transformed_df = load_enron()
        print(transformed_df.head())
    """

    # Read the Parquet file
    df = pd.read_parquet('test-00000-of-00001.parquet')
    # fulldata = df.to_dict(orient='records') # DEBUG: Convert DataFrame to list of dictionaries (records format)


    #### - CONVERT DATA TO REGAS EXPECTED INPUT - ####
    transformed_data = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        # Skip records with empty or null values in 'alternate_answers'
        if len(row['alternate_answers']) == 0 or len(row['alternate_answers'][0]) == 0:
            continue

        # Create the transformed row
        transformed_row = {
            'question': row['questions'][0],          # Map 'question' to 'questions'
            'answer': row['alternate_answers'][0][0], # Map 'answer' to the first 'alternate_answers'
            'contexts': [row['email']],              # Map 'contexts' to 'email'
            'ground_truth': (row['gold_answers'])[0]    # Map 'ground_truth' to 'gold_answers'
        }
        transformed_data.append(transformed_row)

    # Create the transformed DataFrame
    transformed_df = pd.DataFrame(transformed_data)
    return transformed_df