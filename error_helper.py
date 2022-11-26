"""
import error_helper

CHECKPOINT_PATH='./densenet121_baseline_best_fold0_2a1ca668-540d-11ed-abb4-aa0cb2e0f96a.pth'

error_helper.get_model_results(f'./{CHECKPOINT_PATH[2:-4]}_train_results_tilted.csv')
"""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    fbeta_score,
    recall_score,
    precision_score,
)
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from glob import glob

pd.set_option("mode.chained_assignment", None)


# Helper functions
def define_eval_metrics(df):
    recall = np.round(recall_score(y_true=df.actual, y_pred=df.fractured) * 100, 2)
    precision = np.round(
        precision_score(y_true=df.actual, y_pred=df.fractured) * 100, 2
    )

    tn, fp, fn, tp = confusion_matrix(y_true=df.actual, y_pred=df.fractured).ravel()

    fpr = np.round(fp / (fp + tn) * 100, 2)
    fnr = np.round(fn / (fn + tp) * 100, 2)

    accuracy = np.round(
        accuracy_score(
            y_true=df.actual,
            y_pred=df.fractured,
        )
        * 100
    )

    f1 = np.round(fbeta_score(y_true=df.actual, y_pred=df.fractured, beta=1) * 100, 2)

    f05 = np.round(
        fbeta_score(y_true=df.actual, y_pred=df.fractured, beta=0.5) * 100, 2
    )

    f2 = np.round(fbeta_score(y_true=df.actual, y_pred=df.fractured, beta=2) * 100, 2)

    return [recall, precision, tn, fp, fn, tp, fpr, fnr, accuracy, f1, f05, f2]


def eval_model(df):
    """
    Need fractured, row_id, actual columns
    """
    # try:
    #     print(
    #         f"Average inference time : {inference_time/(len(df)/8):.3f} s per patient"
    #     )
    # except:
    #     print("No time data available")

    # initialize patient and vertebrae df
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
        "accuracy",
        "f1",
        "f05",
        "f2",
    ]

    # get eval metrics at overall patient and overall vertebrae
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
                "accuracy",
                "f1",
                "f05",
                "f2",
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
                "accuracy",
                "f1",
                "f05",
                "f2",
            ]
        ),
        ["patient_level", "vertebrae_level", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    ].applymap(
        "{:.2f}%".format
    )

    return output


def format_vertebrae_analysis(model_results):
    """
    Format model results to just look at vertebrae in this format:
           Unnamed: 0                     row_id  actual  fractured vertebrae actual_vertebrae predicted_vertebrae
               0  1.2.826.0.1.3680043.10001       0          0        C1             None                None
               1  1.2.826.0.1.3680043.10001       0          0        C2             None                None
               2  1.2.826.0.1.3680043.10001       0          0        C3             None                None
    """
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
        model_results_vert["fractured"] == 1,
        "predicted_vertebrae",
    ] = model_results_vert.loc[model_results_vert["fractured"] == 1, "vertebrae"]

    output_fname = "fmt_raw_vertebrae.csv"
    model_results_vert.to_csv(output_fname)
    print(f"Evaluation results stored in {output_fname}")
    return model_results_vert


def get_vertebrae_crosstab(model_results):
    model_results_vert = format_vertebrae_analysis(model_results)

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

    return (
        crosstab_predict[["actual_vertebrae", "predicted_vertebrae"]]
        .value_counts(dropna=False, normalize=True)
        .map(lambda x: "{:.2f}%".format(x * 100))
        .reset_index(name="proportion")
    )


def get_shifts_in_predictions(model_results):
    model_results_vert = format_vertebrae_analysis(model_results=model_results)

    # Consider only matches where there was a fracture and the model detected it
    crosstab_lst = pd.merge(
        model_results_vert[model_results_vert.actual_vertebrae.notnull()]
        .groupby("row_id")
        .agg({"actual_vertebrae": list})
        .reset_index(),
        model_results_vert[model_results_vert.predicted_vertebrae.notnull()]
        .groupby("row_id")
        .agg({"predicted_vertebrae": list})
        .reset_index(),
        how="inner",
    )

    # Are inferior vertebrae always predicted?
    # ex. predicts c7 instead of c1
    crosstab_lst["all_pred_inferior"] = crosstab_lst.apply(
        lambda row: all(
            actual_el < pred_el
            for actual_el in row["actual_vertebrae"]
            for pred_el in row["predicted_vertebrae"]
        ),
        axis=1,
    )
    print(crosstab_lst["all_pred_inferior"].value_counts(normalize=True))

    crosstab_lst["any_pred_inferior"] = crosstab_lst.apply(
        lambda row: any(
            actual_el < pred_el
            for actual_el in row["actual_vertebrae"]
            for pred_el in row["predicted_vertebrae"]
        ),
        axis=1,
    )
    print(crosstab_lst["any_pred_inferior"].value_counts(normalize=True))

    # Are superior vertebrae always predicted?
    # ex. predicts c1 instead of c7
    crosstab_lst["all_pred_superior"] = crosstab_lst.apply(
        lambda row: all(
            actual_el > pred_el
            for actual_el in row["actual_vertebrae"]
            for pred_el in row["predicted_vertebrae"]
        ),
        axis=1,
    )
    print(crosstab_lst["all_pred_superior"].value_counts(normalize=True))

    crosstab_lst["any_pred_superior"] = crosstab_lst.apply(
        lambda row: any(
            actual_el > pred_el
            for actual_el in row["actual_vertebrae"]
            for pred_el in row["predicted_vertebrae"]
        ),
        axis=1,
    )
    print(crosstab_lst["any_pred_superior"].value_counts(normalize=True))


def get_fracture_counts(model_results):
    model_results_vert = format_vertebrae_analysis(model_results=model_results)

    # Consider only matches where there was a fracture and the model detected it
    crosstab_lst = pd.merge(
        model_results_vert[model_results_vert.actual_vertebrae.notnull()]
        .groupby("row_id")
        .agg({"actual_vertebrae": list})
        .reset_index(),
        model_results_vert[model_results_vert.predicted_vertebrae.notnull()]
        .groupby("row_id")
        .agg({"predicted_vertebrae": list})
        .reset_index(),
        how="outer",
    )
    crosstab_lst.loc[
        ~crosstab_lst.actual_vertebrae.isna(), "actual_vertebrae"
    ] = crosstab_lst.loc[
        ~crosstab_lst.actual_vertebrae.isna(), "actual_vertebrae"
    ].apply(
        lambda x: len(x)
    )
    crosstab_lst.loc[
        ~crosstab_lst.predicted_vertebrae.isna(), "predicted_vertebrae"
    ] = crosstab_lst.loc[
        ~crosstab_lst.predicted_vertebrae.isna(), "predicted_vertebrae"
    ].apply(
        lambda x: len(x)
    )
    crosstab_lst = crosstab_lst.fillna(0)

    for el in range(crosstab_lst["actual_vertebrae"].nunique()):
        print(f"\n\nProcessing {el} actual fractures...")
        print(
            crosstab_lst.loc[
                crosstab_lst.actual_vertebrae == el, "predicted_vertebrae"
            ].describe()
        )
        


def get_sagittal_view(bad_patients):
    # visualize the sagittal vies

    # load images from input
    sagittal_images = "/root/input/rsna-2022-cervical-spine-fracture-detection/sagittal_train_images/*.png"
    random_images = random.choices(
        [
            image
            for image in glob(sagittal_images)
            if image.split("/")[-1][:-4] in bad_patients
        ],
        k=9,
    )

    sagittal_patient_id = []
    for i in random_images:
        sagittal_patient_id.append(i.split(".")[-2])

    # Get images
    images = [mpimg.imread(image) for image in random_images]

    # Plot images
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 12))
    fig.suptitle(
        f"Sagittal View - Randomly Pick 9 Bad Predictions", weight="bold", size=20
    )

    start = 0
    for i in range(start, start + 9):
        img = images[i]
        patient_id = sagittal_patient_id[i]
        # Plot the image
        x = (i - start) // 3
        y = (i - start) % 3

        axes[x, y].imshow(img, cmap="bone")
        axes[x, y].set_title(f"patient_id: {patient_id}", fontsize=14, weight="bold")
        axes[x, y].axis("off")


def get_worst_patients(model_results):
    # use competition loss function to define worst patients?
    model_results_vert = model_results[
        ~model_results.row_id.str.contains("patient_overall")
    ].copy()
    model_results_vert["row_id"] = model_results_vert.row_id.apply(
        lambda x: x.split("_")[0]
    )

    bad_patients = model_results_vert.loc[
        (model_results_vert.actual == 1) & (model_results_vert.fractured == 0), "row_id"
    ].unique()

    get_sagittal_view(bad_patients)


# Get overall model performance
def get_model_results(fname, verbose=True):

    # CHECKPOINT_PATH='./densenet121_baseline_best_fold0_2a1ca668-540d-11ed-abb4-aa0cb2e0f96a.pth'
    # model_results = pd.read_csv(f'./{CHECKPOINT_PATH[2:-4]}_train_results.csv')

    # TODO debug fname = "full_model_validation/valid_folds_all_fmt.csv"
    model_results = pd.read_csv(fname)

    if verbose:
        print(eval_model(model_results))
        print(get_vertebrae_crosstab(model_results))
        print(get_shifts_in_predictions(model_results))

    output_fname = "eval_model.csv"
    eval_model(model_results).to_csv(output_fname)
    print(f"Evaluation results stored in {output_fname}")

    output_fname = "vertebrae_crosstab.csv"
    get_vertebrae_crosstab(model_results).to_csv(output_fname)
    print(f"Vertebrae crosstab results stored in {output_fname}")

    try:
        get_worst_patients(model_results)
    except Exception as e:
        print("Error with worst patients code")
        print(e)
