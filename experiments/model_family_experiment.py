"""One fixed CatBoost/LightGBM comparison; no test evaluation or parameter search."""
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from importlib.metadata import version
import hashlib
import json
import pickle
import platform

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, confusion_matrix)
import mlflow
import mlflow.sklearn
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXPERIMENT = "churn-xgboost-weighted-improvement"
BASELINE_RUN = "e7a79bcc5ff94e1b82147a937abd357b"
BASELINE_PATH = ROOT / "experiments/mlruns/final_random_oversampling/models/m-dfa4a9fcaca844d5bf6cd1eb55020dca/artifacts"
BASELINE_THRESHOLD = 0.5112516283988953


class AveragePrecisionMetric:
    """Unweighted AP on the original distribution, even with class-weighted training."""
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        return float(average_precision_score(np.asarray(target), expit(np.asarray(approxes[0])))), 1.0

    def get_final_error(self, error, weight):
        return error


def lightgbm_ap(y_true, probabilities):
    return "average_precision", float(average_precision_score(y_true, probabilities)), True


def evaluate(y, probabilities, threshold):
    predicted = probabilities >= threshold
    return {"pr_auc": average_precision_score(y, probabilities),
        "f1": f1_score(y, predicted, zero_division=0),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "accuracy": accuracy_score(y, predicted)}


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    table = pd.DataFrame({"threshold": thresholds, "f1": f1, "precision": precision, "recall": recall})
    table = table.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True])
    return float(table.iloc[0].threshold), table


def prepare_data(output):
    # Reproduce data_analysis.ipynb, including its earlier in-place dropna cell.
    df = pd.read_csv(ROOT / "data/customer_churn_1M.csv").dropna().drop_duplicates()
    clean = df.drop(columns=["customer_id"], errors="ignore")
    clean = clean[clean.gender.astype("string").str.strip().str.lower().ne("other")].copy()
    clean["churn"] = clean.churn.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "yes": 1, "1": 1, "false": 0, "no": 0, "0": 0}
    clean = clean[clean.churn.isin(mapping)].copy()
    clean["churn"] = clean.churn.map(mapping).astype(int)
    clean = clean.drop_duplicates().reset_index(drop=True)
    all_indices = np.arange(len(clean))
    train_ids, holdout_ids = train_test_split(all_indices, test_size=0.2, random_state=SEED, stratify=clean.churn)
    val_ids, _ = train_test_split(holdout_ids, test_size=0.5, random_state=SEED,
                                stratify=clean.churn.iloc[holdout_ids])
    raw_train = clean.iloc[train_ids].drop(columns="churn").copy()
    raw_val = clean.iloc[val_ids].drop(columns="churn").copy()
    y_train, y_val = clean.churn.iloc[train_ids].to_numpy(), clean.churn.iloc[val_ids].to_numpy()
    with np.load(ROOT / "data/processed_datasets.npz", allow_pickle=False) as data:
        X_train, X_val = data["X_train"], data["X_val"]
        np.testing.assert_array_equal(y_train, data["Y_train"])
        np.testing.assert_array_equal(y_val, data["Y_val"])
    categories = raw_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = raw_train.select_dtypes(exclude=["object", "string"]).columns.tolist()
    frequency_columns = [col for col in categories if raw_train[col].nunique(dropna=False) > 100]
    one_hot_columns = [col for col in categories if col not in frequency_columns]
    engineered_train, engineered_val = raw_train.copy(), raw_val.copy()
    mappings = {}
    for col in frequency_columns:
        mappings[col] = raw_train[col].value_counts(dropna=False, normalize=True)
        engineered_train[col] = raw_train[col].map(mappings[col]).fillna(0)
        engineered_val[col] = raw_val[col].map(mappings[col]).fillna(0)
    preprocessor = ColumnTransformer([
        ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())]), numeric + frequency_columns),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), one_hot_columns),
    ], remainder="drop")
    normalized = MinMaxScaler()
    reproduced_train = normalized.fit_transform(preprocessor.fit_transform(engineered_train))
    reproduced_val = normalized.transform(preprocessor.transform(engineered_val))
    # Exact float32 reproduction proves both row order and preprocessing agree with saved arrays.
    np.testing.assert_array_equal(reproduced_train.astype("float32"), X_train)
    np.testing.assert_array_equal(reproduced_val.astype("float32"), X_val)
    del reproduced_train, reproduced_val, engineered_train, engineered_val, df, clean
    np.savez_compressed(output / "split_indices.npz", train=train_ids, validation=val_ids)
    with (output / "engineered_preprocessing.pkl").open("wb") as stream:
        pickle.dump({"frequency_mappings": mappings, "preprocessor": preprocessor,
                     "normalizer": normalized}, stream)
    for col in categories:
        # No learned encoding or manual one-hot encoding: preserve original category strings.
        raw_train[col] = raw_train[col].astype(str)
        raw_val[col] = raw_val[col].astype(str)
    schema = {"raw_columns": raw_train.columns.tolist(), "categorical_columns": categories,
        "categorical_cardinality_train": {c: int(raw_train[c].nunique()) for c in categories},
        "train_rows": len(y_train), "validation_rows": len(y_val),
        "train_churn": int(y_train.sum()), "validation_churn": int(y_val.sum()),
        "engineered_features": X_train.shape[1], "exact_preprocessing_reproduction": True,
        "split_seed": SEED, "test_arrays_loaded": False,
        "split_indices_sha256": hashlib.sha256(train_ids.tobytes() + val_ids.tobytes()).hexdigest()}
    (output / "data_schema.json").write_text(json.dumps(schema, indent=2))
    print("Verified exact training/validation labels and engineered arrays.", flush=True)
    print("Native categorical columns:", categories, flush=True)
    print("Categorical cardinalities:", schema["categorical_cardinality_train"], flush=True)
    return X_train, X_val, y_train, y_val, raw_train, raw_val, categories


def paired_uncertainty(y, candidate_scores, baseline_scores, candidate_threshold, baseline_threshold):
    """Fixed-prediction stratified bootstrap; no model or threshold fitting."""
    rng = np.random.default_rng(SEED)
    classes = [np.flatnonzero(y == k) for k in [0, 1]]
    ap_deltas, f1_deltas = [], []
    for _ in range(200):
        idx = np.concatenate([rng.choice(ids, size=len(ids), replace=True) for ids in classes])
        ap_deltas.append(average_precision_score(y[idx], candidate_scores[idx]) -
                         average_precision_score(y[idx], baseline_scores[idx]))
        f1_deltas.append(f1_score(y[idx], candidate_scores[idx] >= candidate_threshold) -
                         f1_score(y[idx], baseline_scores[idx] >= baseline_threshold))
    return {"paired_bootstrap_ap_delta_95_interval": np.quantile(ap_deltas, [0.025, 0.975]).tolist(),
            "paired_bootstrap_f1_delta_95_interval": np.quantile(f1_deltas, [0.025, 0.975]).tolist()}


def main():
    search_id = uuid4().hex
    output = ROOT / "data/model_family_comparison" / search_id
    output.mkdir(parents=True)
    print("Output:", output, flush=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'experiments/mlflow_churn.db'}")
    mlflow.set_experiment(EXPERIMENT)
    X_train, X_val, y_train, y_val, raw_train, raw_val, categories = prepare_data(output)
    predictions, models, rows = {}, {}, []
    environment = {"python": platform.python_version(), **{p: version(p) for p in
        ["catboost", "lightgbm", "xgboost", "scikit-learn", "numpy", "pandas", "mlflow"]}}

    def record(name, model, val_X, fit_seconds, params, model_kind, run, historic=False):
        scores = model.predict_proba(val_X)[:, 1]
        threshold, thresholds = choose_threshold(y_val, scores)
        metric_values = evaluate(y_val, scores, threshold)
        if name == "xgboost_baseline":
            np.testing.assert_allclose(metric_values["pr_auc"], 0.20672509732768157, atol=1e-10, rtol=0)
            np.testing.assert_allclose(threshold, BASELINE_THRESHOLD, atol=1e-10, rtol=0)
        run_dir = output / name
        run_dir.mkdir(exist_ok=True)
        thresholds.to_csv(run_dir / "validation_thresholds.csv", index=False)
        np.savez_compressed(run_dir / "validation_predictions.npz", y=y_val, probabilities=scores)
        cm = confusion_matrix(y_val, scores >= threshold, labels=[0, 1])
        pd.DataFrame(cm, index=["actual_no_churn", "actual_churn"],
            columns=["predicted_no_churn", "predicted_churn"]).to_csv(run_dir / "confusion_matrix.csv")
        mlflow.log_params({**params, "decision_threshold": threshold,
                          "fit_time_source": "historical_source_run" if historic else "this_run"})
        mlflow.log_metrics({**metric_values, "fit_seconds": fit_seconds})
        mlflow.log_dict(environment, "environment.json")
        mlflow.log_dict({"threshold": threshold, "rule": "predict_proba(X)[:, 1] >= threshold",
                        "input": "native categorical dataframe" if model_kind == "catboost" else "engineered numeric features"},
                       "prediction_settings.json")
        mlflow.log_artifacts(str(run_dir), artifact_path="validation")
        mlflow.log_artifact(str(output / "data_schema.json"))
        mlflow.log_artifact(str(output / "split_indices.npz"))
        mlflow.log_artifact(str(output / "engineered_preprocessing.pkl"))
        if model_kind == "catboost":
            model_info = mlflow.catboost.log_model(model, name="model")
            restored = mlflow.catboost.load_model(model_info.model_uri)
        else:
            model_info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            restored = mlflow.sklearn.load_model(model_info.model_uri)
        subset = val_X.iloc[:128] if isinstance(val_X, pd.DataFrame) else val_X[:128]
        np.testing.assert_allclose(restored.predict_proba(subset)[:, 1], scores[:128])
        row = {"model": name, "threshold": threshold, **metric_values,
               "fit_seconds": fit_seconds, "fit_time_historical": historic,
               "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
               "run_id": run.info.run_id, "model_uri": model_info.model_uri}
        rows.append(row); predictions[name] = scores; models[name] = model
        pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
        print(name, json.dumps(row), flush=True)

    def start(name):
        return mlflow.start_run(run_name="model_family_" + name, tags={
            "search_id": search_id, "experiment_type": "final_model_family",
            "evaluation_split": "validation", "primary_metric": "unweighted_average_precision",
            "baseline_source_run_id": BASELINE_RUN})

    with start("xgboost_baseline") as run:
        baseline = mlflow.sklearn.load_model(str(BASELINE_PATH))
        source = mlflow.get_run(BASELINE_RUN)
        baseline_params = baseline.named_steps["model"].get_params()
        record("xgboost_baseline", baseline, X_val, source.data.metrics["fit_seconds"],
               {**{k: baseline_params[k] for k in ["learning_rate", "max_depth", "n_estimators",
                  "subsample", "colsample_bytree", "scale_pos_weight"]}, "n_features": 25,
                "retrained": False, "original_tree_budget": 2000, "effective_trees": 445},
               "sklearn", run, historic=True)

    # Exactly one fixed LightGBM configuration. No resampling, feature engineering, or search.
    lgb_params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31,
        max_depth=-1, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, class_weight={0: 1, 1: 7}, random_state=SEED,
        n_jobs=8, verbosity=-1, metric="None", deterministic=True, force_col_wise=True)
    with start("lightgbm") as run:
        model = lgb.LGBMClassifier(**lgb_params)
        started = perf_counter()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=lightgbm_ap,
                  callbacks=[lgb.early_stopping(50, first_metric_only=True), lgb.log_evaluation(100)])
        elapsed = perf_counter() - started
        mlflow.log_dict(model.evals_result_, "learning_curve.json")
        record("lightgbm", model, X_val, elapsed, {**lgb_params,
               "n_features": X_train.shape[1], "effective_trees": model.best_iteration_,
               "early_stopping_rounds": 50}, "sklearn", run)

    cat_params = dict(iterations=2000, learning_rate=0.05, depth=6,
        loss_function="Logloss", class_weights=[1, 7], random_seed=SEED,
        thread_count=8, l2_leaf_reg=3, allow_writing_files=False,
        one_hot_max_size=1, early_stopping_rounds=50, use_best_model=True)
    with start("catboost") as run:
        model = CatBoostClassifier(**cat_params, eval_metric=AveragePrecisionMetric())
        started = perf_counter()
        model.fit(Pool(raw_train, y_train, cat_features=categories),
                  eval_set=Pool(raw_val, y_val, cat_features=categories), verbose=100)
        elapsed = perf_counter() - started
        mlflow.log_dict(model.get_evals_result(), "learning_curve.json")
        record("catboost", model, raw_val, elapsed, {**cat_params,
               "categorical_columns": categories, "n_features": raw_train.shape[1],
               "eval_metric": "unweighted_average_precision", "effective_trees": model.tree_count_},
               "catboost", run)

    # Predeclared ensemble gate: strong ranking plus sufficiently different residuals.
    # Test at most one equal-weight blend; do not tune weights or search subsets.
    diagnostics = []
    for name in ["catboost", "lightgbm"]:
        residual_correlation = float(np.corrcoef(predictions[name] - y_val,
                                                predictions["xgboost_baseline"] - y_val)[0, 1])
        ap = next(r["pr_auc"] for r in rows if r["model"] == name)
        diagnostics.append({"model": name, "residual_correlation": residual_correlation,
            "pr_auc": ap, "eligible": residual_correlation < 0.95 and ap >= rows[0]["pr_auc"]})
    eligible = [d for d in diagnostics if d["eligible"]]
    ensemble_note = "Skipped: no candidate met both AP >= baseline and residual correlation < 0.95."
    if eligible:
        partner = max(eligible, key=lambda d: d["pr_auc"])["model"]
        name = "ensemble_xgboost_" + partner
        scores = (predictions["xgboost_baseline"] + predictions[partner]) / 2
        threshold, thresholds = choose_threshold(y_val, scores)
        values = evaluate(y_val, scores, threshold)
        cm = confusion_matrix(y_val, scores >= threshold, labels=[0, 1])
        with start(name) as run:
            mlflow.log_params({"members": ["xgboost_baseline", partner], "weights": [0.5, 0.5],
                              "threshold": threshold, "weight_search": False})
            mlflow.log_metrics(values)
            # Member models plus fixed composition recipe are the ensemble artifact.
            recipe = {"members": {r["model"]: r["model_uri"] for r in rows
                                  if r["model"] in ["xgboost_baseline", partner]},
                      "weights": [0.5, 0.5], "threshold": threshold,
                      "operation": "arithmetic mean of positive-class predict_proba",
                      "inputs": {"xgboost_baseline": "engineered numeric features",
                                 partner: "native categorical dataframe" if partner == "catboost" else "engineered numeric features"}}
            mlflow.log_dict(recipe, "ensemble_recipe.json")
            np.savez_compressed(output / "ensemble_validation_predictions.npz", y=y_val, probabilities=scores)
            mlflow.log_artifact(str(output / "ensemble_validation_predictions.npz"))
            thresholds.to_csv(output / "ensemble_thresholds.csv", index=False)
            mlflow.log_artifact(str(output / "ensemble_thresholds.csv"))
            mlflow.log_dict({"matrix": cm.tolist(), "labels": [0, 1]}, "confusion_matrix.json")
            rows.append({"model": name, "threshold": threshold, **values,
                "fit_seconds": 0.0, "fit_time_historical": False, "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]), "run_id": run.info.run_id,
                "model_uri": "runs:/" + run.info.run_id + "/ensemble_recipe.json"})
        predictions[name] = scores
        ensemble_note = "Evaluated one equal-weight blend with " + partner + "; no weight tuning."

    comparison = pd.DataFrame(rows).sort_values(["pr_auc", "f1"], ascending=False)
    for key in ["pr_auc", "f1", "precision", "recall", "roc_auc", "balanced_accuracy", "accuracy"]:
        comparison[key + "_delta_vs_baseline"] = comparison[key] - rows[0][key]
    comparison.to_csv(output / "comparison.csv", index=False)
    best_ap = comparison.iloc[0]
    best_f1 = comparison.sort_values(["f1", "pr_auc"], ascending=False).iloc[0]
    # Fixed-prediction paired bootstrap, not another model/threshold search.
    # Describes sampling uncertainty only; it cannot correct validation-selection optimism.
    best_new = comparison[comparison["model"].isin(["catboost", "lightgbm"])].iloc[0]
    uncertainty = paired_uncertainty(y_val, predictions[best_new["model"]],
        predictions["xgboost_baseline"], float(best_new["threshold"]), float(rows[0]["threshold"]))
    summary = {"search_id": search_id, "experiment": EXPERIMENT,
        "best_pr_auc": best_ap["model"], "best_f1": best_f1["model"],
        "best_pr_auc_delta": float(best_ap["pr_auc_delta_vs_baseline"]),
        "bootstrap_candidate": best_new["model"],
        **uncertainty,
        "bootstrap_replicates": 200, "ensemble_diagnostics": diagnostics, "ensemble_note": ensemble_note,
        "test_evaluated": False, "historical_test_inspection": True,
        "limitations": ["Single fixed configuration and seed per new family; no hyperparameter search.",
          "CatBoost uses original categorical values; LightGBM uses 44 engineered features; XGBoost uses top 25. This compares candidate pipelines, not isolated algorithms.",
          "Early stopping, threshold selection, and comparison reuse validation data; bootstrap intervals do not remove selection optimism.",
          "Baseline training time comes from its historical run; no retraining performed.",
          "Original preprocessing removes missing rows and excludes gender='other'; comparisons apply only to that retained population.",
          "Test split was inspected for an earlier baseline; it was not accessed in this experiment."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    with start("comparison") as run:
        mlflow.log_artifact(str(output / "comparison.csv"))
        mlflow.log_artifact(str(output / "summary.json"))
        mlflow.log_artifact(str(Path(__file__)), artifact_path="source")
        mlflow.log_artifact(str(ROOT / "experiments/model_family_comparison.ipynb"), artifact_path="source")
        mlflow.log_params({"best_pr_auc": best_ap["model"], "best_f1": best_f1["model"]})
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return output


if __name__ == "__main__":
    main()
