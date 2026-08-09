"""
Rebuilds outputs/interactive/dashboard.html by inlining Chart.js and the
data JSON into dashboard_template.html.
"""

import os

BUILD_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BUILD_DIR, "..", "dashboard.html")


def build():
    with open(os.path.join(BUILD_DIR, "dashboard_template.html")) as f:
        template = f.read()
    with open(os.path.join(BUILD_DIR, "chart.umd.min.js")) as f:
        chartjs = f.read()
    with open(os.path.join(BUILD_DIR, "dashboard_data.json")) as f:
        data_json = f.read()

    out = template.replace("/*!CHARTJS_LIB*/", chartjs).replace("/*!DATA_JSON*/", data_json)

    with open(OUTPUT_PATH, "w") as f:
        f.write(out)

    print(f"Saved -> {os.path.abspath(OUTPUT_PATH)} ({len(out):,} bytes)")


if __name__ == "__main__":
    build()
