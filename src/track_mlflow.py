import mlflow

mlflow.set_experiment("rul-prediction")

with mlflow.start_run(run_name="lstm-fd001"):
    mlflow.log_params({
        "dataset": "FD001",
        "model": "LSTM",
        "seq_len": 30,
        "hidden": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "rul_cap": 125,
    })

    mlflow.log_metrics({
        "rmse": 13.11,
        "nasa_score": 275.5,
    })

    mlflow.log_artifact("models/lstm.pt")
    mlflow.log_artifact("models/scaler.joblib")

print("Kayıt tamam")