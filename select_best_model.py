import shutil

# choose the file you want as production model:
best_model_path = "models/global_best_model_optuna.pkl"

shutil.copy(best_model_path, "models/best_model.pkl")
print("Copied best model to models/best_model.pkl")
