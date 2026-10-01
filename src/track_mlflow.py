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

    # BURAYA README'deki gerçek değerlerini yaz
    mlflow.log_metrics({
        "rmse": 0.0,
        "nasa_score": 0.0,
    })

    mlflow.log_artifact("models/lstm.pt")
    mlflow.log_artifact("models/scaler.joblib")

print("Kayıt tamam")