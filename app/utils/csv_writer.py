import os
import pandas as pd


def append_result(result, output_path):
    """Append one experiment result to the CSV.

    If the output CSV already exists, we append without writing headers.
    To avoid missing new columns (e.g., timing/evaluation fields), we align
    the appended row to the existing CSV schema.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    new_df = pd.DataFrame([result])

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, nrows=0)
        existing_cols = list(existing_df.columns)

        # Ensure all existing columns exist in new row
        for c in existing_cols:
            if c not in new_df.columns:
                new_df[c] = ""

        # Only write columns that exist in the file (same order)
        new_df = new_df[existing_cols]

        new_df.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False
        )

    else:
        new_df.to_csv(
            output_path,
            mode="w",
            header=True,
            index=False
        )
