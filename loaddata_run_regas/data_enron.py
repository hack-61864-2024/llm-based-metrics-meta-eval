import pandas as pd

def load_enron():
    #### - IMPORT THE DATA - ####
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