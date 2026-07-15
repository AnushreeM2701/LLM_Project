import pandas as pd
from openpyxl.styles import Alignment
from openpyxl import load_workbook


def csv_to_excel(csv_path):

    excel_path = csv_path.replace(".csv", ".xlsx")

    df = pd.read_csv(csv_path)

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Experiment Results"
        )

        worksheet = writer.sheets["Experiment Results"]

        worksheet.freeze_panes = "A2"

        worksheet.auto_filter.ref = worksheet.dimensions

        widths = {
            "A": 25,
            "B": 15,
            "C": 18,
            "D": 15,
            "E": 12,
            "F": 12,
            "G": 12,
            "H": 70,
            "I": 20,
            "J": 70,
            "K": 70,
            "L": 20,
            "M": 15,
            "N": 15,
            "O": 20,
            "P": 20,
            "Q": 18
        }

        for col, width in widths.items():

            worksheet.column_dimensions[col].width = width

        for row in worksheet.iter_rows():

            max_lines = 1

            for cell in row:

                cell.alignment = Alignment(
                    wrap_text=True,
                    vertical="top"
                )

                if cell.value:

                    max_lines = max(
                        max_lines,
                        str(cell.value).count("\n") + 1
                    )

            worksheet.row_dimensions[row[0].row].height = max(
                25,
                max_lines * 18
            )

    print()
    print("=" * 60)
    print("Excel file created")
    print(excel_path)
    print("=" * 60)