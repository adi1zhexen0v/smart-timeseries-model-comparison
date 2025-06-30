import os
import pandas as pd

def merge_traffic_csv_files(input_dir: str) -> pd.DataFrame:
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    merged_df = pd.concat(
        [pd.read_csv(os.path.join(input_dir, file)) for file in all_files],
        ignore_index=True
    )
    return merged_df
