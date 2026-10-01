import numpy as np
import torch
import torch.nn as nn
from features import prepare
from metrics import nasa_score

torch.manual_seed(42)
np.random.seed(42)

Y_SCALE = 125.0

# 1. Veriyi al (etiketler 0-1 arasına çekilir)
X_train, y_train, X_test, y_test = prepare()
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train) / Y_SCALE
X_test = torch.tensor(X_test)

# 2. Model
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

model = LSTMModel(n_features=X_train.shape[2])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Eğitim
EPOCHS = 40
BATCH = 64

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(X_train))
    for i in range(0, len(X_train), BATCH):
        idx = perm[i:i + BATCH]
        optimizer.zero_grad()
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy() * Y_SCALE
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    print(f"Epoch {epoch:2d}  Test RMSE: {rmse:.2f}")

print("LSTM son RMSE:", round(float(rmse), 2))
print("LSTM NASA skoru:", round(nasa_score(y_test, pred), 1))