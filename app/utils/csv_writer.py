import os
import pandas as pd


def append_result(result, output_path):
    """
    Append one experiment result to CSV.

    Creates the CSV if it doesn't exist.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df = pd.DataFrame([result])

    if os.path.exists(output_path):

        df.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            output_path,
            mode="w",
            header=True,
            index=False
        )