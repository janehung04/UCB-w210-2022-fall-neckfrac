import Evaluation.error_helper as error_helper
import pandas as pd

import importlib

importlib.reload(Evaluation.error_helper)


if __name__ == "__main__":
    fname = "ensemble_overall_f1"
    df = pd.read_csv(f"Evaluation/ensemble_model/{fname}.csv")

    df["row_id"] = df.StudyInstanceUID + "_" + df.prediction_type

    df = df[["row_id", "label", "ensemble_label"]].rename(
        columns={"ensemble_label": "fractured", "label": "actual"}
    )

    df.to_csv(f"Evaluation/ensemble_model/{fname}_fmt.csv")

    # Get model results
    error_helper.get_model_results(
        f"Evaluation/ensemble_model/{fname}_fmt.csv", verbose=True
    )
