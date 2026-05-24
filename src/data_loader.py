import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the insurance dataset from a pipe-delimited text file.

    Args:
        filepath: Path to the raw data file.

    Returns:
        A pandas DataFrame with the loaded data.
    """

    try:
        df = pd.read_csv(filepath, sep="|", low_memory=False)

        if df.empty:
            raise ValueError("The dataset is empty.")

        print(
            f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns"
        )

        return df

    except FileNotFoundError:
        print(f"Error: File not found -> {filepath}")

    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")

    except pd.errors.ParserError:
        print("Error: Failed to parse the file. Check the delimiter or file format.")

    except Exception as e:
        print(f"Unexpected error while loading data: {e}")

    return pd.DataFrame()


def get_basic_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.

    Args:
        df: The loaded DataFrame.
    """

    try:
        if df.empty:
            print("Error: DataFrame is empty.")
            return

        print("\n--- Shape ---")
        print(df.shape)

        print("\n--- Data Types ---")
        print(df.dtypes)

        print("\n--- Missing Values ---")
        print(df.isnull().sum())

        print("\n--- First 5 Rows ---")
        print(df.head())

    except Exception as e:
        print(f"Unexpected error while displaying dataset info: {e}")