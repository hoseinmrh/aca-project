import json
import pandas as pd

def compute_linear_score(normalized_metrics):
    weights = {
    "BalancedAccuracy": 0.4,
    "TrainingTimePerEpoch": 0.15,
    "PeakMemoryMB": 0.25,
    "ModelSizeMB": 0.1,
    "InferenceLatencyMS": 0.1
    }
    score = (
        weights["BalancedAccuracy"] * normalized_metrics["BalancedAccuracy"]
        + weights["TrainingTimePerEpoch"] * (1 / (normalized_metrics["TrainingTimePerEpoch"] + 1))
        + weights["PeakMemoryMB"] * (1 / (normalized_metrics["PeakMemoryMB"] + 1))
        + weights["ModelSizeMB"] * (1 / (normalized_metrics["ModelSizeMB"] + 1))
        + weights["InferenceLatencyMS"] * (1 / (normalized_metrics["InferenceLatencyMS"] + 1))
    )
    return score



def main():
    models = ["enhanced_gnn", "graph_transformer", "HGP_SL"]
    metric_keys = ["BalancedAccuracy", "TrainingTimePerEpoch", "PeakMemoryMB", "ModelSizeMB", "InferenceLatencyMS"]
    rows = []
    rows_2 = []
    for model in models:
        metrics_path = f"../{model}/metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        row = {k: metrics[k] for k in metric_keys}
        row_2 = metrics
        # Round the values of row_2 to 3 decimal places
        for k in row_2:
            row_2[k] = round(row_2[k], 3)
        row_2["Model"] = model
        rows_2.append(row_2)
        row["Model"] = model
        rows.append(row)

    df = pd.DataFrame(rows)
    df_2 = pd.DataFrame(rows_2)
    df_2.to_csv("metrics.csv", index=False)
    # Normalize the metrics columns (min-max normalization)
    for k in metric_keys:
        min_val = df[k].min()
        max_val = df[k].max()
        if max_val > min_val:
            df[k] = (df[k] - min_val) / (max_val - min_val)
        else:
            df[k] = 0.0  # or 1.0, but all values are the same
    
    print(df)

    # Compute the score for each row using normalized metrics
    df["Score"] = df.apply(lambda row: compute_linear_score({k: row[k] for k in metric_keys}), axis=1)

    # Save only model name and score
    df[["Model", "Score"]].to_csv("combined_metrics.csv", index=False)

if __name__ == "__main__":
    main()



