import os, glob, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import joblib

RAW_GLOB = "cicddos/raw/*.csv"  
OUT_DIR = "cicddos/processed"
ART_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

TARGET = "label"
TEST_SIZE = 0.2
RANDOM_STATE = 42

RENAME_MAP = {
    "Flow ID": "flow_id",
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
    "Timestamp": "timestamp",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Bwd IAT Std": "bwd_iat_std",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_pkts_per_s",
    "Label": TARGET
}

ID_LIKE = ["flow_id", "src_ip", "dst_ip", "timestamp"]

def to_snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_per_")

def coerce_numeric(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan

def safe_div(a, b):
    try:
        a = 0.0 if pd.isna(a) else float(a)
        b = 0.0 if pd.isna(b) else float(b)
        if b == 0.0:
            return 0.0
        return a / b
    except Exception:
        return 0.0

def detect_duration_seconds(series):
    med = series.median()
    if pd.isna(med):
        return series
    if med > 1e6:
        return series / 1e6
    if med > 1e3:
        return series / 1e3
    return series

def map_label_binary(x):
    s = str(x).lower().strip()
    return 0 if "benign" in s else 1

def renyi_entropy(values, alpha=2.0, bins=16):
    vals = pd.Series(values).dropna()
    if vals.empty:
        return 0.0
    cats = pd.cut(vals, bins=bins, include_lowest=True)
    counts = cats.value_counts().astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    if abs(alpha - 1.0) < 1e-12:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    s = float((p ** alpha).sum())
    return float((1.0 / (1.0 - alpha)) * np.log(s + 1e-12))

def load_concat(globpat):
    files = sorted(glob.glob(globpat))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {globpat}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        new_cols = {}
        for c in df.columns:
            c_strip = c.strip()
            new_cols[c] = RENAME_MAP.get(c_strip, to_snake(c_strip))
        df = df.rename(columns=new_cols)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def build_features_light(df):
    df = df.copy()

    # limpiar numéricos
    for c in df.columns:
        if c in [TARGET, "timestamp", "src_ip", "dst_ip", "flow_id"]:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].apply(coerce_numeric)
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # duración
    if "flow_duration" in df.columns:
        df["flow_duration_s"] = detect_duration_seconds(pd.to_numeric(df["flow_duration"], errors="coerce"))
    else:
        df["flow_duration_s"] = np.nan

    # totales
    if {"total_fwd_packets","total_bwd_packets"}.issubset(df.columns):
        df["total_packets"] = df["total_fwd_packets"].fillna(0) + df["total_bwd_packets"].fillna(0)
    elif "flow_pkts_per_s" in df.columns and "flow_duration_s" in df.columns:
        df["total_packets"] = df["flow_pkts_per_s"].fillna(0) * df["flow_duration_s"].fillna(0)
    else:
        df["total_packets"] = np.nan

    if {"total_fwd_bytes","total_bwd_bytes"}.issubset(df.columns):
        df["total_bytes"] = df["total_fwd_bytes"].fillna(0) + df["total_bwd_bytes"].fillna(0)
    elif "flow_bytes_per_s" in df.columns and "flow_duration_s" in df.columns:
        df["total_bytes"] = df["flow_bytes_per_s"].fillna(0) * df["flow_duration_s"].fillna(0)
    else:
        df["total_bytes"] = np.nan

    # tasas
    df["pps"] = df.apply(lambda r: safe_div(r.get("total_packets", 0.0), r.get("flow_duration_s", 0.0)), axis=1)
    df["bps"] = df.apply(lambda r: safe_div(8.0 * r.get("total_bytes", 0.0), r.get("flow_duration_s", 0.0)), axis=1)

    # ratios
    if {"total_fwd_packets","total_bwd_packets"}.issubset(df.columns):
        df["ratio_fwd_bwd_pkts"] = df.apply(lambda r: safe_div(r.get("total_fwd_packets",0.0), r.get("total_bwd_packets",0.0)), axis=1)
    if {"total_fwd_bytes","total_bwd_bytes"}.issubset(df.columns):
        df["ratio_fwd_bwd_bytes"] = df.apply(lambda r: safe_div(r.get("total_fwd_bytes",0.0), r.get("total_bwd_bytes",0.0)), axis=1)

    # IAT combinados
    iat_cols = [c for c in ["fwd_iat_mean","bwd_iat_mean","flow_iat_mean"] if c in df.columns]
    if iat_cols:
        df["iat_mean"] = df[iat_cols].mean(axis=1, numeric_only=True)
    iat_std_cols = [c for c in ["fwd_iat_std","bwd_iat_std","flow_iat_std"] if c in df.columns]
    if iat_std_cols:
        df["iat_std"] = df[iat_std_cols].mean(axis=1, numeric_only=True)

    # pkt_len_mean
    if {"fwd_pkt_len_mean","bwd_pkt_len_mean"}.issubset(df.columns):
        df["pkt_len_mean"] = df[["fwd_pkt_len_mean","bwd_pkt_len_mean"]].mean(axis=1, numeric_only=True)

    # entropía por src_ip 
    if "src_ip" in df.columns and "src_port" in df.columns:
        df["H_src_port_renyi2"] = df.groupby("src_ip")["src_port"].transform(
            lambda s: renyi_entropy(s.values, alpha=2.0, bins=16)
        )
    if "src_ip" in df.columns and "dst_port" in df.columns:
        df["H_dst_port_renyi2"] = df.groupby("src_ip")["dst_port"].transform(
            lambda s: renyi_entropy(s.values, alpha=2.0, bins=16)
        )
    if "src_ip" in df.columns and "pkt_len_mean" in df.columns:
        df["H_pktlen_renyi2"] = df.groupby("src_ip")["pkt_len_mean"].transform(
            lambda s: renyi_entropy(s.values, alpha=2.0, bins=16)
        )

    return df

def select_and_split_light(df):
    meta_cols = [c for c in ID_LIKE if c in df.columns]
    meta = df[meta_cols + [TARGET]].copy() if meta_cols else pd.DataFrame({TARGET: df[TARGET]})

    df[TARGET] = df[TARGET].apply(map_label_binary).astype(int)

    # filtro low-rate
    if "pps" in df.columns:
        df = df[df["pps"] < 200].copy()
        print(f"Filtro low-rate aplicado (pps < 200), registros: {len(df)}")

    X = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore").copy()

    for c in X.columns:
        if c == TARGET:
            continue
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].apply(coerce_numeric)

    nunique = X.drop(columns=[TARGET]).nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep + [TARGET]]

    train_df, test_df = train_test_split(X, test_size=TEST_SIZE, stratify=X[TARGET], random_state=RANDOM_STATE)

    feats = [c for c in train_df.columns if c != TARGET]
    prep = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("sc", MinMaxScaler())
    ])

    Xtr = prep.fit_transform(train_df[feats])
    Xte = prep.transform(test_df[feats])

    kept_mask = prep.named_steps["var"].get_support()
    kept_cols = [f for f, k in zip(feats, kept_mask) if k]

    train_out = pd.DataFrame(Xtr, columns=kept_cols)
    train_out[TARGET] = train_df[TARGET].values

    test_out = pd.DataFrame(Xte, columns=kept_cols)
    test_out[TARGET] = test_df[TARGET].values

    train_out.to_csv(os.path.join(OUT_DIR, "train_lowrate.csv"), index=False)
    test_out.to_csv(os.path.join(OUT_DIR, "test_lowrate.csv"), index=False)
    meta_train = meta.loc[train_df.index]
    meta_test  = meta.loc[test_df.index]
    meta_train.to_csv(os.path.join(OUT_DIR, "meta_train.csv"), index=False)
    meta_test.to_csv(os.path.join(OUT_DIR, "meta_test.csv"), index=False)

    joblib.dump(prep, os.path.join(ART_DIR, "prep_lowrate_pipeline_light.joblib"))
    with open(os.path.join(ART_DIR, "feature_names_lowrate_light.json"), "w", encoding="utf-8") as f:
        json.dump({"features_selected": kept_cols, "target": TARGET}, f, indent=2)

    print("OK ->", os.path.join(OUT_DIR, "train_lowrate.csv"))
    print("OK ->", os.path.join(OUT_DIR, "test_lowrate.csv"))
    print("Train balance:", train_out[TARGET].value_counts().to_dict())
    print("Test balance:", test_out[TARGET].value_counts().to_dict())
    print("N features:", len(kept_cols))

def main():
    raw = load_concat(RAW_GLOB)
    if TARGET not in raw.columns:
        raise ValueError("No hay columna 'label' en los CSV")
    df = build_features_light(raw)
    select_and_split_light(df)

if __name__ == "__main__":
    main()
