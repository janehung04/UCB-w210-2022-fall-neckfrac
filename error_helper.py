"""
import error_helper

CHECKPOINT_PATH='./densenet121_baseline_best_fold0_2a1ca668-540d-11ed-abb4-aa0cb2e0f96a.pth'

error_helper.get_model_results(f'./{CHECKPOINT_PATH[2:-4]}_train_results_tilted.csv')
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

pd.set_option("mode.chained_assignment", None)


# Helper functions
def define_eval_metrics(df):
    cls_report = classification_report(
        y_true=df.actual, y_pred=df.fractured, zero_division=0, output_dict=True
    )

    recall = np.round(cls_report["1"]["recall"] * 100)
    precision = np.round(cls_report["1"]["precision"] * 100)

    tn, fp, fn, tp = confusion_matrix(y_true=df.actual, y_pred=df.fractured).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    pred_pos_rate = np.round(np.mean(df.fractured == 1) * 100)
    actual_pos_rate = np.round(np.mean(df.actual == 1) * 100)

    return [recall, precision, tn, fp, fn, tp, fpr, fnr]


def eval_model(df):

    try:
        print(
            f"Average inference time : {inference_time/(len(df)/8):.3f} s per patient"
        )
    except:
        print("No time data available")

    # initialize patient and vertebrae df
    df["fractured"] = round(df.fractured)
    patient_df = df[df.row_id.str.contains("patient_overall")]
    vert_df = df[~df.row_id.str.contains("patient_overall")]

    # store eval metrics
    eval_metrics = dict()
    eval_metrics["eval_metric"] = [
        "recall",
        "precision",
        "tn",
        "fp",
        "fn",
        "tp",
        "fpr",
        "fnr",
    ]

    eval_metrics["patient_level"] = define_eval_metrics(patient_df)
    eval_metrics["vertebrae_level"] = define_eval_metrics(vert_df)

    # which vertebrae are missed the most? FP FN?
    vert_df.loc[:, "vertebrae"] = vert_df.row_id.str.split(
        "_",
    ).apply(lambda x: x[-1])

    for vertebra in vert_df.vertebrae.unique():
        eval_metrics[vertebra] = define_eval_metrics(
            vert_df.loc[vert_df.vertebrae == vertebra]
        )

    # issues with patient overall not match patient vertebrae prediction?

    output = pd.DataFrame.from_dict(eval_metrics)

    # convert to right format
    output.loc[
        output.eval_metric.isin(
            [
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ),
        ["patient_level", "vertebrae_level", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    ] = output.loc[
        output.eval_metric.isin(
            [
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ),
        ["patient_level", "vertebrae_level", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    ].applymap(
        lambda x: "{:d}".format(int(x))
    )

    output.loc[
        output.eval_metric.isin(
            [
                "recall",
                "precision",
                "fpr",
                "fnr",
            ]
        ),
        ["patient_level", "vertebrae_level", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    ] = output.loc[
        output.eval_metric.isin(
            [
                "recall",
                "precision",
                "fpr",
                "fnr",
            ]
        ),
        ["patient_level", "vertebrae_level", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    ].applymap(
        "{:.2f}%".format
    )

    return output


def get_vertebrae_crosstab(model_results):
    model_results_vert = model_results[
        ~model_results.row_id.str.contains("patient_overall")
    ].copy()
    model_results_vert["vertebrae"] = model_results_vert.row_id.apply(
        lambda x: x.split("_")[-1]
    )
    model_results_vert["row_id"] = model_results_vert.row_id.apply(
        lambda x: x.split("_")[0]
    )

    model_results_vert["actual_vertebrae"] = None
    model_results_vert["predicted_vertebrae"] = None
    model_results_vert.loc[
        model_results_vert["actual"] == 1, "actual_vertebrae"
    ] = model_results_vert.loc[model_results_vert["actual"] == 1, "vertebrae"]
    model_results_vert.loc[
        np.round(model_results_vert["fractured"]) == 1, "predicted_vertebrae"
    ] = model_results_vert.loc[
        np.round(model_results_vert["fractured"]) == 1, "vertebrae"
    ]

    crosstab_predict = pd.merge(
        model_results_vert[model_results_vert.actual_vertebrae.notnull()]
        .groupby("row_id")
        .agg({"actual_vertebrae": lambda x: ",".join(x)})
        .reset_index(),
        model_results_vert[model_results_vert.predicted_vertebrae.notnull()]
        .groupby("row_id")
        .agg({"predicted_vertebrae": lambda x: ",".join(x)})
        .reset_index(),
        how="outer",
    )

    return crosstab_predict[["actual_vertebrae", "predicted_vertebrae"]].value_counts(
        dropna=False, normalize=True
    ).map(lambda x: "{:.2f}%".format(x*100))


# Get overall model performance
def get_model_results(fname):

    # CHECKPOINT_PATH='./densenet121_baseline_best_fold0_2a1ca668-540d-11ed-abb4-aa0cb2e0f96a.pth'
    # model_results = pd.read_csv(f'./{CHECKPOINT_PATH[2:-4]}_train_results.csv')
    model_results = pd.read_csv(fname)

    display(eval_model(model_results))

    display(get_vertebrae_crosstab(model_results))
