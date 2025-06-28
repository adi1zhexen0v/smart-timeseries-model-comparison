import pandas as pd
from typing import List, Tuple, Optional, Dict

def scale_features(df: pd.DataFrame, features: List[str], target: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    scaler_params = {}
    df_scaled = df.copy()

    columns_to_scale = features + ([target] if target else [])

    for col in columns_to_scale:
        mean = df_scaled[col].mean()
        std = df_scaled[col].std()

        if std == 0 or pd.isna(std):
            std = 1.0

        df_scaled[col] = (df_scaled[col] - mean) / std
        scaler_params[col] = {"mean": mean, "std": std}

    return df_scaled, scaler_params
