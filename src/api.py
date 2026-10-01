import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SEQ_LEN = 30

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

bundle = joblib.load("models/scaler.joblib")
scaler, feats = bundle["scaler"], bundle["feats"]

model = LSTMModel(n_features=len(feats))
model.load_state_dict(torch.load("models/lstm.pt"))
model.eval()

app = FastAPI(title="RUL Tahmin API")

class Window(BaseModel):
    window: list[list[float]]

@app.get("/health")
def health():
    return {"status": "ok", "sensors": feats, "seq_len": SEQ_LEN}

@app.post("/predict")
def predict(data: Window):
    arr = np.array(data.window, dtype=np.float32)
    if arr.shape != (SEQ_LEN, len(feats)):
        raise HTTPException(
            status_code=422,
            detail=f"Beklenen boyut: ({SEQ_LEN}, {len(feats)}), gelen: {arr.shape}",
        )
    arr = scaler.transform(arr).astype(np.float32)
    x = torch.tensor(arr).unsqueeze(0)
    with torch.no_grad():
        rul = model(x).item() * 125.0
    return {"rul": round(max(rul, 0.0), 1)}