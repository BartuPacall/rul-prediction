import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

def load_fd(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)

def add_rul(df, cap=125):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df

def constant_cols(df, threshold=0.01):
    feats = [c for c in df.columns if c not in ("unit", "cycle", "RUL")]
    return [c for c in feats if df[c].std() < threshold]

if __name__ == "__main__":
    train = add_rul(load_fd("data/raw/train_FD001.txt"))
    drop = constant_cols(train)
    print("Atılacak sütunlar:", drop)
    train = train.drop(columns=drop)
    print("Yeni boyut:", train.shape)