import error_helper
import pandas as pd

import importlib

importlib.reload(error_helper)


if __name__ == "__main__":
    df = pd.read_csv("ensemble_model/ensemble_overall.csv")

    df["row_id"] = df.StudyInstanceUID + "_" + df.prediction_type

    df = df[["row_id", "label", "ensemble_label"]].rename(
        columns={"ensemble_label": "fractured", "label": "actual"}
    )

    df.to_csv("ensemble_model/ensemble_overall_fmt.csv")

    # Get model results
    error_helper.get_model_results(
        "ensemble_model/ensemble_overall_fmt.csv", verbose=True
    )
