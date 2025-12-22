import pandas as pd
from dagshub import dagshub_logger

METRICS_PATH = "metrics/all_experiments.csv"

def main():
    df = pd.read_csv(METRICS_PATH)

    for _, row in df.iterrows():
        exp_name = f"exp_{int(row['experiment_id'])}"
        print(f"Logging {exp_name} ...")

        with dagshub_logger() as logger:
            # treat experiment name as a hyperparameter so you can see it
            logger.log_hyperparams(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "experiment_name": exp_name,
                    "model": row["model"],
                    "pca_applied": bool(row["pca_applied"]),
                    "hyperparameter_tuning": bool(row["hyperparameter_tuning"]),
                }
            )
            logger.log_metrics(
                {
                    "f1_score": float(row["f1_score"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "roc_auc": float(row["roc_auc"]),
                }
            )

        print(f"Done {exp_name}")

if __name__ == "__main__":
    main()
