import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
print(f"Downloaded to: {path}")
