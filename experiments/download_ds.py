import kagglehub

# Download latest version
path = kagglehub.dataset_download("isandeep06/customer-churn-prediction-dataset-1m")

print("Path to dataset files:", path)