import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from data import load_fd, add_rul, constant_cols

# 1. Eğitim verisini hazırla
train = add_rul(load_fd("data/raw/train_FD001.txt"))
drop = constant_cols(train)
train = train.drop(columns=drop)
feats = [c for c in train.columns if c not in ("unit", "cycle", "RUL")]

# 2. Ölçekle (scaler sadece eğitim verisine öğretilir)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train[feats])
y_train = train["RUL"]

# 3. Test verisini hazırla (her motorun SON satırı alınır)
test = load_fd("data/raw/test_FD001.txt").drop(columns=drop)
last = test.groupby("unit").tail(1)
X_test = scaler.transform(last[feats])
y_test = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].clip(upper=125)

# 4. Modeli eğit
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Sonuç
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Random Forest RMSE:", round(rmse, 2))


# 6. XGBoost
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
print("XGBoost RMSE:", round(rmse_xgb, 2))