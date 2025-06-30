import pandas as pd
from tqdm import tqdm

def fill_missing_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    full_dates = df['date'].unique()
    full_sites = df['site_id'].unique()

    filled_data = []

    progress_bar = tqdm(total=len(full_dates) * len(full_sites), desc="Imputing missing data")

    for date in full_dates:
        for site in full_sites:
            site_day_data = df[(df['date'] == date) & (df['site_id'] == site)]

            if site_day_data.shape[0] == 2:
                filled_data.append(site_day_data)
            else:
                times = sorted(site_day_data['timestamp'].dt.time.unique())
                if len(times) == 1:
                    timestamps = [pd.Timestamp.combine(pd.Timestamp(date), times[0]), pd.Timestamp.combine(pd.Timestamp(date), times[0])]
                else:
                    timestamps = pd.date_range(start=pd.Timestamp(date), periods=2, freq="12h")

                site_subset = df[df['site_id'] == site]
                site_avg = site_subset.mean(numeric_only=True)

                for ts in timestamps:
                    if not any(abs((site_day_data['timestamp'] - ts).dt.total_seconds()) < 60):
                        new_row = {
                            'timestamp': ts,
                            'site_id': site,
                            'lat': site_subset['lat'].mode().iloc[0] if not site_subset['lat'].isna().all() else None,
                            'long': site_subset['long'].mode().iloc[0] if not site_subset['long'].isna().all() else None,
                            **{col: site_avg[col] for col in site_avg.index if col not in ['lat', 'long']}
                        }
                        site_day_data = pd.concat([site_day_data, pd.DataFrame([new_row])], ignore_index=True)

                filled_data.append(site_day_data)

            progress_bar.update(1)

    progress_bar.close()

    final_df = pd.concat(filled_data, ignore_index=True)
    final_df = final_df.sort_values(by=["timestamp", "site_id"]).reset_index(drop=True)
    return final_df
