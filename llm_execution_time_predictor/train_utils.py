from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import json
from typing import List, cast
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import lightgbm as lgb
from onnxmltools.convert import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
import os
import onnxruntime as ort

def preprocess_input_for_prediction(
    batch_size, avg_context_len, gpu, mode="prefill"
) -> List[float]:
    if mode == "prefill":
        num_new_tokens = batch_size * avg_context_len
        prod_ext_ctx = batch_size * (avg_context_len**2)
        num_context_tokens = avg_context_len * batch_size
        num_batch_size = batch_size
    else:
        num_new_tokens = batch_size
        prod_ext_ctx = batch_size * avg_context_len
        num_context_tokens = avg_context_len * batch_size
        num_batch_size = batch_size
    return [num_new_tokens, prod_ext_ctx, num_context_tokens, num_batch_size]


def _percentile_from_sorted(sorted_arr, q):
    """
    Pre Sorts the percentiles for perf
    """
    n = sorted_arr.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_arr[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    if lo == hi:
        return float(sorted_arr[lo])
    frac = rank - lo
    return float((1.0 - frac) * sorted_arr[lo] + frac * sorted_arr[hi])

def preprocess_input_considering_seq_and_cached_len(seq_lens, cached_context_lens, stage):
    """
    Build a single-sample rich feature vector for prediction without pandas.
    """
    if stage not in ("prefill", "decode"):
        raise ValueError("stage must be either 'prefill' or 'decode'")
    if len(seq_lens) != len(cached_context_lens):
        raise ValueError("seq_lens and cached_context_lens must have the same length")

    seq = np.asarray(seq_lens, dtype=np.float64)
    cached = np.asarray(cached_context_lens, dtype=np.float64)
    extend = np.maximum(0.0, seq - cached)

    batch_size = float(seq.size)
    total_token_length = float(np.sum(seq))

    seq_sorted = np.sort(seq)
    len_max = float(seq_sorted[-1])
    len_min = float(seq_sorted[0])
    len_std = float(np.std(seq_sorted))
    len_p90 = _percentile_from_sorted(seq_sorted, 90)
    len_p95 = _percentile_from_sorted(seq_sorted, 95)

    cached_sum = float(np.sum(cached))
    cached_max = float(np.max(cached))
    cached_ratio = cached_sum / max(1.0, total_token_length)

    extend_sum = float(np.sum(extend))
    extend_sorted = np.sort(extend)
    extend_max = float(extend_sorted[-1])
    extend_mean = float(np.mean(extend_sorted))
    extend_std = float(np.std(extend_sorted))
    extend_p90 = _percentile_from_sorted(extend_sorted, 90)

    imbalance = (len_max / len_min) if len_min > 0 else np.nan

    if stage == "prefill":
        num_new_tokens = extend_sum
        prod_ext_ctx = float(batch_size * (len_max ** 2))
    else:
        num_new_tokens = batch_size
        prod_ext_ctx = float(batch_size * len_max)
    num_context_tokens = float(batch_size * len_max)

    len_mean = float(np.mean(seq_sorted))
    mid = seq_sorted.size // 2
    if seq_sorted.size % 2 == 1:
        len_median = float(seq_sorted[mid])
    else:
        len_median = float((seq_sorted[mid - 1] + seq_sorted[mid]) / 2.0)
    len_range = len_max - len_min
    len_p99 = _percentile_from_sorted(seq_sorted, 99)
    len_cv = len_std / max(1.0, len_mean)

    extend_min = float(extend_sorted[0])
    mid_e = extend_sorted.size // 2
    if extend_sorted.size % 2 == 1:
        extend_median = float(extend_sorted[mid_e])
    else:
        extend_median = float((extend_sorted[mid_e - 1] + extend_sorted[mid_e]) / 2.0)
    extend_p99 = _percentile_from_sorted(extend_sorted, 99)
    extend_cv = extend_std / max(1.0, extend_mean) if extend_mean != 0.0 else np.nan

    prompt_ratio = extend_sum / max(1.0, total_token_length)
    cached_peak_ratio = cached_max / max(1.0, len_max)
    B_len_mean = float(batch_size * len_mean)
    B_len_max_sq = float(batch_size * (len_max ** 2))

    # Keep placeholders consistent with training
    skew = np.nan
    cache_percent = np.nan
    cache_len_prod = cache_percent * len_max if not np.isnan(cache_percent) else np.nan

    log_len_max = float(np.log1p(len_max))
    log_prod_ext_ctx = float(np.log1p(prod_ext_ctx))
    log_num_context_tokens = float(np.log1p(num_context_tokens))

    values = [
        num_new_tokens, prod_ext_ctx, num_context_tokens,
        len_max, len_min, len_std, len_p90, len_p95,
        cached_sum, cached_max, cached_ratio,
        extend_max, extend_mean, extend_std, extend_p90,
        batch_size, imbalance, skew,
        cache_percent,
        len_mean, len_median, len_range, len_p99, len_cv,
        extend_min, extend_median, extend_p99, extend_cv,
        prompt_ratio, cached_peak_ratio, B_len_mean, B_len_max_sq, cache_len_prod,
        log_len_max, log_prod_ext_ctx, log_num_context_tokens,
    ]
    return np.asarray(values, dtype=np.float32)

def build_stage_features(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    Build input features for latency modeling based on the inference stage.
    Returns
    -------
    pd.DataFrame
        A dataframe with engineered features:
        - num_new_tokens: total tokens processed/generated (models token compute)
        - prod_ext_ctx: proxy for attention cost (quadratic or linear depending on stage)
        - num_context_tokens: total context tokens active (models memory + cache pressure)
        - batch_size: degree of parallelism
        - time: latency target to be predicted
    """
    df = df.copy()
    # TODO: integrate the actual df["batch_lens"] to make a better predictor. Possibly the max?
    if stage == "prefill":
        # Each request has `input_len` tokens; all tokens are processed in parallel
        # Attention complexity is O(seq_len^2) per request
        df["num_new_tokens"] = df["batch_size"] * df["input_len"]
        df["prod_ext_ctx"] = df["batch_size"] * (df["input_len"] ** 2)
        df["num_context_tokens"] = df["batch_size"] * df["input_len"]
        df["time"] = df["latency"]

    elif stage == "decode":
        # One token is generated per request per step
        # Each new token attends to all previous context (linear in output_len)
        df["num_new_tokens"] = df["batch_size"]
        df["prod_ext_ctx"] = df["batch_size"] * df["input_len"]
        df["num_context_tokens"] = df["batch_size"] * df["input_len"]
        df["time"] = df["latency"]
    else:
        raise ValueError("stage must be either 'prefill' or 'decode'")

    return df[
        ["num_new_tokens", "prod_ext_ctx", "num_context_tokens", "batch_size", "time"]
    ]


def train_linear_predictor(train_df: pd.DataFrame, name):
    """
    Train a linear regression model to predict latency based on engineered features.
    """
    X_train = train_df[
        ["num_new_tokens", "prod_ext_ctx", "num_context_tokens", "batch_size"]
    ].to_numpy(dtype=np.float32)
    y_train = train_df["time"].to_numpy(dtype=np.float32)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_train)

    print(f"Linear Regression: {name}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_lr)) * 1000:.2f}ms")
    print(f"Train MAE: {mean_absolute_error(y_train, y_pred_lr) * 1000:.2f}ms")
    print(f"Train R2: {r2_score(y_train, y_pred_lr):.4f}")
    return lr_model


def train_tree_predictor(train_df: pd.DataFrame, name):
    """
    Train a decision tree model to predict latency based on engineered features.
    """

    # Extract features and target
    X_train = train_df[
        ["num_new_tokens", "prod_ext_ctx", "num_context_tokens", "batch_size"]
    ].to_numpy(dtype=np.float32)
    y_train = train_df["time"].to_numpy(dtype=np.float32)

    # Fit Decision Tree Regressor
    tree_model = RandomForestRegressor(
        n_estimators=10, random_state=42, min_samples_leaf=2, max_depth=12
    )
    tree_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_tree = tree_model.predict(X_train)

    print(f"Decision Tree: {name}")
    print(
        f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_tree)) * 1000:.2f}ms"
    )
    print(f"Train MAE: {mean_absolute_error(y_train, y_pred_tree) * 1000:.2f}ms")
    print(f"Train R2: {r2_score(y_train, y_pred_tree):.4f}")
    return tree_model

def _as_list(x):
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

def batch_size_regime(batch_size):
    if batch_size < 30:
        return 0
    elif batch_size < 100:
        return 1
    else:
        return 2

def seq_len_regime(seq_len):
    if seq_len < 512:
        return 0
    elif seq_len < 2048:
        return 1
    elif seq_len < 8192:
        return 2
    else:
        return 3
    
def build_stage_features_rich(df, stage):
    if stage not in ("prefill", "decode"):
        raise ValueError("stage must be either 'prefill' or 'decode'")
    df = df.copy()
    for c in ["combined_seq_lens", "cached_prefix_lens", "new_extend_lens"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v) if isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]") else v)
    if "total_extend_len" in df.columns and "new_extend_lens" in df.columns:
        df["total_extend_len"] = df["total_extend_len"].fillna(df["new_extend_lens"].apply(lambda xs: sum(xs) if isinstance(xs, list) else xs))
    if "cache_percent" not in df.columns:
        df["cache_percent"] = np.nan
    df["len_max"] = df["combined_seq_lens"].apply(lambda x: np.max(_as_list(x)))
    df["len_min"] = df["combined_seq_lens"].apply(lambda x: np.min(_as_list(x)))
    df["len_std"] = df["combined_seq_lens"].apply(lambda x: np.std(_as_list(x)))
    df["len_p90"] = df["combined_seq_lens"].apply(lambda x: np.percentile(_as_list(x), 90))
    df["len_p95"] = df["combined_seq_lens"].apply(lambda x: np.percentile(_as_list(x), 95))
    df["cached_sum"] = df["cached_prefix_lens"].apply(lambda x: np.sum(_as_list(x)))
    df["cached_max"] = df["cached_prefix_lens"].apply(lambda x: np.max(_as_list(x)))
    df["cached_ratio"] = df["cached_sum"] / df["total_token_length"].clip(lower=1)
    df["extend_sum"] = df["new_extend_lens"].apply(lambda x: np.sum(_as_list(x)))
    df["extend_max"] = df["new_extend_lens"].apply(lambda x: np.max(_as_list(x)))
    df["extend_mean"] = df["new_extend_lens"].apply(lambda x: np.mean(_as_list(x)))
    df["extend_std"] = df["new_extend_lens"].apply(lambda x: np.std(_as_list(x)))
    df["extend_p90"] = df["new_extend_lens"].apply(lambda x: np.percentile(_as_list(x), 90))
    df["imbalance"] = df["len_max"] / df["len_min"].replace(0, np.nan)
    if stage == "prefill":
        df["num_new_tokens"] = df["extend_sum"]
        df["prod_ext_ctx"] = df["batch_size"] * (df["len_max"] ** 2)
    else:
        df["num_new_tokens"] = df["batch_size"]
        df["prod_ext_ctx"] = df["batch_size"] * df["len_max"]
    df["num_context_tokens"] = df["batch_size"] * df["len_max"]
    df["time"] = df["latency"]
    df["len_mean"] = df["combined_seq_lens"].apply(lambda x: np.mean(_as_list(x)))
    df["len_median"] = df["combined_seq_lens"].apply(lambda x: np.median(_as_list(x)))
    df["len_range"] = df["len_max"] - df["len_min"]
    df["len_p99"] = df["combined_seq_lens"].apply(lambda x: np.percentile(_as_list(x), 99))
    df["len_cv"] = df["len_std"] / df["len_mean"].clip(lower=1)
    df["extend_min"] = df["new_extend_lens"].apply(lambda x: np.min(_as_list(x)))
    df["extend_median"] = df["new_extend_lens"].apply(lambda x: np.median(_as_list(x)))
    df["extend_p99"] = df["new_extend_lens"].apply(lambda x: np.percentile(_as_list(x), 99))
    df["extend_cv"] = df["extend_std"] / df["extend_mean"].clip(lower=1)
    df["prompt_ratio"] = df["extend_sum"] / df["total_token_length"].clip(lower=1)
    df["cached_peak_ratio"] = df["cached_max"] / df["len_max"].clip(lower=1)
    df["B_len_mean"] = df["batch_size"] * df["len_mean"]
    df["B_len_max_sq"] = df["batch_size"] * (df["len_max"] ** 2)
    df["cache_len_prod"] = df["cache_percent"] * df["len_max"]
    
    for col in ["prod_ext_ctx", "num_context_tokens"]:
        df[f"log_{col}"] = np.log1p(df[col])
    df["normalized_skew"] = df["len_std"] / df["len_mean"].clip(lower=1)
    df["extend_skew"] = df["extend_std"] / df["extend_mean"].clip(lower=1)
    df["batch_size_regime"] = df["batch_size"].apply(batch_size_regime)
    df["seq_len_regime"] = df["len_max"].apply(seq_len_regime)
    

    keep = [
        "num_new_tokens","prod_ext_ctx","num_context_tokens",
        "len_max","len_min","len_std","len_p90","len_p95",
        "cached_sum","cached_max","cached_ratio",
        "extend_max","extend_mean","extend_std","extend_p90",
        "batch_size","imbalance","normalized_skew","extend_skew",
        "batch_size_regime","seq_len_regime",
        "cache_percent",
        "len_mean","len_median","len_range","len_p99","len_cv",
        "extend_min","extend_median","extend_p99","extend_cv",
        "prompt_ratio","cached_peak_ratio","B_len_mean","B_len_max_sq","cache_len_prod",
        "time",
    ]
    return df[keep]

def train_lgbm_predictor(train_df, stage, n_estimators=300, num_leaves=64, learning_rate=0.05, min_data_in_leaf=10, test_size=0.01, random_state=42):
    X = train_df.drop(columns=["time"])
    y = train_df["time"]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf, random_state=random_state, verbose=-1)
    model.fit(X_tr, y_tr)
    def _m(Xs, ys):
        p = model.predict(Xs)
        return float(np.sqrt(mean_squared_error(ys, p)) * 1000.0), float(mean_absolute_error(ys, p) * 1000.0), float(r2_score(ys, p))
    tr_rmse, tr_mae, tr_r2 = _m(X_tr, y_tr)
    va_rmse, va_mae, va_r2 = _m(X_val, y_val)
    metrics = {
        "stage": stage,
        "train_rmse_ms": tr_rmse,
        "train_mae_ms": tr_mae,
        "train_r2": tr_r2,
        "val_rmse_ms": va_rmse,
        "val_mae_ms": va_mae,
        "val_r2": va_r2,
        "n_features": X.shape[1],
        "feature_columns": list(X.columns),
    }
    return model, metrics

def export_lgbm_to_onnx(model, n_features, out_path, input_name="input"):
    onnx_model = convert_lightgbm(model, initial_types=[(input_name, FloatTensorType([None, n_features]))])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

def load_onyx_model(path):
    session = ort.InferenceSession(path)
    return session

def predict_lgbm_onyx(model, seq_lens, cached_context_lens, stage):
    x = preprocess_input_considering_seq_and_cached_len(seq_lens, cached_context_lens, stage)[None, :]
    try:
        input_name = model.get_inputs()[0].name
    except Exception:
        input_name = "input"
    y = model.run(None, {input_name: x})[0]
    return float(y[0])
