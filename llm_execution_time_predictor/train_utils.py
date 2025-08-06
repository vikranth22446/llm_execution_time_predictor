from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import json


def preprocess_input_for_prediction(
    batch_size, avg_context_len, gpu, mode="prefill"
) -> float:
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
