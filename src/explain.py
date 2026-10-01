import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from data import load_fd, add_rul, constant_cols

# 1. Veriyi hazırla (train_baseline.py ile aynı adımlar)
train = add_rul(load_fd("data/raw/train_FD001.txt"))
drop = constant_cols(train)
train = train.drop(columns=drop)
feats = [c for c in train.columns if c not in ("unit", "cycle", "RUL")]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train[feats])
y_train = train["RUL"]

# 2. XGBoost'u eğit
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 3. SHAP değerlerini hesapla (hız için 2000 örnek)
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=2000, replace=False)
X_sample = X_train[idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 4. Grafiği kaydet
shap.summary_plot(shap_values, X_sample, feature_names=feats, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=150)
print("Kaydedildi: reports/shap_summary.png")

# 5. Sensör önem sırası
importance = np.abs(shap_values).mean(axis=0)
for name, val in sorted(zip(feats, importance), key=lambda x: -x[1]):
    print(f"{name}: {val:.2f}")