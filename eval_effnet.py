import error_helper
import pandas as pd

import importlib

importlib.reload(error_helper)



if __name__ == "__main__":
    df = pd.read_csv("effnet_models/effnet_inference_500_patients.csv")

    df = df[["row_id", "actual", "pred_flag"]].rename({"pred_flag": "fractured"})
    
    df.to_csv("effnet_models/effnet_inference_500_patients_fmt.csv")

    # Get model results
    error_helper.get_model_results(
        "effnet_models/effnet_inference_500_patients_fmt.csv", verbose=True
    )
