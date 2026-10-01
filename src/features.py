import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data import load_fd, add_rul, constant_cols

SEQ_LEN = 30

def prepare(seq_len=SEQ_LEN):
    # Eğitim verisi
    train = add_rul(load_fd("data/raw/train_FD001.txt"))
    drop = constant_cols(train)
    train = train.drop(columns=drop)
    feats = [c for c in train.columns if c not in ("unit", "cycle", "RUL")]

    scaler = MinMaxScaler()
    train[feats] = scaler.fit_transform(train[feats])

    # Her motor için 30 satırlık kayan pencereler
    X_train, y_train = [], []
    for _, g in train.groupby("unit"):
        values = g[feats].values
        ruls = g["RUL"].values
        for i in range(len(g) - seq_len + 1):
            X_train.append(values[i:i + seq_len])
            y_train.append(ruls[i + seq_len - 1])

    # Test verisi: her motorun son 30 çevrimi
    test = load_fd("data/raw/test_FD001.txt").drop(columns=drop)
    test[feats] = scaler.transform(test[feats])
    X_test = [g[feats].values[-seq_len:] for _, g in test.groupby("unit")]
    y_test = pd.read_csv("data/raw/RUL_FD001.txt", header=None)[0].clip(upper=125)

    return (
        np.array(X_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        np.array(X_test, dtype=np.float32),
        y_test.values.astype(np.float32),
    )

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare()
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)