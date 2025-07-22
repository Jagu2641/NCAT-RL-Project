import pandas as pd

def load_question_bank(csv_path='question_bank.csv'):
    """
    Loads the question bank from a CSV file.
    Args:
        csv_path (str): Path to the question bank CSV file.
    Returns:
        pd.DataFrame: DataFrame with columns ['id', 'question', 'difficulty']
    """
    df = pd.read_csv(csv_path)
    # Ensure correct columns and types
    df['difficulty'] = df['difficulty'].astype(float)
    return df 