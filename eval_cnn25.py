import error_helper
import pandas as pd
import numpy as np

import importlib
importlib.reload(error_helper)


def get_eval_model(df_iter, **kwargs):
    # Get patient overall predicted and actuals
    patient_overall = pd.merge(
        (df_iter.groupby("row_id")["fractured"].max() == 1)
        .astype(int)
        .reset_index()
        .copy(),
        df_iter.groupby("row_id")["actual"].max().reset_index().copy(),
        on="row_id",
        how="inner",
    )
    patient_overall["row_id"] = patient_overall["row_id"] + "_patient_overall"

    # Reformat for use in error analysis code
    df_iter["row_id"] = df_iter.row_id + "_C" + df_iter.cid.astype(str)
    df_iter = pd.concat(
        [df_iter[["row_id", "actual", "fractured"]], patient_overall], axis=0
    )
    df_iter.to_csv("full_model_validation/valid_folds_all_fmt.csv")

    # Get model results
    error_helper.get_model_results(
        "full_model_validation/valid_folds_all_fmt.csv", verbose=kwargs["verbose"]
    )


# Identify the best classification threshold for each vertebrae
def get_best_classification(vert):
    global threshold, best_threshold
    best_f2 = 0.0
    for curr_threshold in np.linspace(0.15, 0.5, 10):

        # For each vertebrae, try a new classification threshold
        threshold.loc[threshold.vert == vert, "threshold"] = curr_threshold

        # Apply new threshold to each vertebrae
        threshold["vert"] = threshold["vert"].astype(int)
        df_iter = df.merge(threshold, left_on="cid", right_on="vert", how="left")
        df_iter["fractured"] = (df_iter.fractured > df_iter.threshold).astype(int)

        ## Call helper
        get_eval_model(df_iter, verbose=False)
        eval_model = pd.read_csv(
            "./eval_model.csv",
        )

        curr_f2 = eval_model.loc[eval_model.eval_metric == "f2", "C" + str(vert)]
        curr_f2 = float(curr_f2.values[0][:-1])

        # Check if the model results are better than what we've stored
        if curr_f2 > best_f2:
            best_f2 = curr_f2
            best_threshold.loc[
                best_threshold.vert == vert, "threshold"
            ] = curr_threshold
            print(f"Best threshold is {curr_threshold} for C{vert}. F2 is {curr_f2}")


def apply_best_threshold(best_threshold):
    best_df = df.merge(best_threshold, left_on="cid", right_on="vert", how="left")
    best_df["fractured"] = (best_df.fractured > best_df.threshold).astype(int)

    # Get patient overall predicted and actuals
    get_eval_model(best_df, verbose=True)


if __name__ == "__main__":
    df = pd.read_csv("full_model_validation/valid_folds_all.csv")

    # Define thresholds for [C1, C2, C3, C4, C5, C6, C7]
    threshold = pd.DataFrame({"vert": range(1, 8), "threshold": [0.5] * 7})
    best_threshold = pd.DataFrame({"vert": range(1, 8), "threshold": [0.5] * 7})

    for vert in range(1, 8):
        get_best_classification(vert=vert)

    print(best_threshold)

    apply_best_threshold(best_threshold)