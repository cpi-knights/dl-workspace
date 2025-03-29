import glob
import gzip
import pandas as pd
from pathlib import Path


def collect_data(dir: str | Path, output_file: str | Path | None = None) -> pd.DataFrame:
    gzipped_files = glob.glob(f'{dir}/**/*.csv.gz', recursive=True)
    print(f"Found {len(gzipped_files)} files")

    dataframes = []

    for file in gzipped_files:
        with gzip.open(file, 'rt') as f:
            df = pd.read_csv(f)
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Found {combined_df.shape[0]} records")
    print(f"Columns: {combined_df.columns.values.tolist()}")

    df_pivot = combined_df.pivot_table(
        index="datetime", columns="parameter", values="value", aggfunc="first"
    ).reset_index()
    df_pivot.rename(columns={"o3": "o3 (ppm)",
                    "pm25": "pm2.5 (µg/m³)"}, inplace=True)

    if output_file:
        df_pivot.to_csv(output_file, index=False, header=True)

    return df_pivot


def preprocess_data(df: pd.DataFrame, output_file: str | Path | None = None) -> pd.DataFrame:
    date_time = pd.to_datetime(df.pop('datetime'), format='ISO8601')

    for col in df.columns.values.tolist():
        col_data = df[col]
        col_data[col_data.isna()] = 0.001
        bad_val = col_data <= 0
        col_data[bad_val] = 0.001
        bad_val = col_data >= 800.0
        col_data[bad_val] = 800.0

    df = df.assign(datetime=date_time)

    if output_file:
        df.to_csv(output_file, index=False, header=True)

    return df
