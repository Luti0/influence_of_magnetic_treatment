import pandas as pd
import sys

def file_to_pd(file_path: str) -> pd.DataFrame:
    """
    File path to DataFrame pandas.
    Columns renaming in DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        new_columns = ['Bm', 'k', 'l', 'r', 'J']
        if len(df.columns) == len(new_columns):
            df.columns = new_columns
        else:
            print(f"Warning: The number of columns in the file ({len(df.columns)}) does not match the expected number ({len(new_columns)}). Columns were not renamed.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
