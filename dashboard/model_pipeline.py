"""Reproduce the WIA1006 notebook's model pipeline for the dashboard's Model tab.

This ports the preprocessing, feature engineering, model bake-off and evaluation
from ``WIA1006_Popularity_Prediction_Final.ipynb`` so the dashboard can show the
project's real results, computed from the source dataset, rather than the
hand-crafted scoring demo used elsewhere in the app.

The training is wrapped in ``st.cache_data`` so the (slow) bake-off runs once per
dataset. ``_train_and_evaluate_impl`` is kept undecorated so it can be exercised
headless, without a Streamlit script context.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	auc,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --- Constants lifted verbatim from the notebook (cells 6-41) --------------------

COLUMN_RENAMES = {
	"app_usage_time_min": "app_usage_min",
	"swipe_right_ratio": "swipe_ratio",
}

COLS_TO_DROP = [
	"interest_tags",         # Free text (hard for ML to read)
	"app_usage_time_label",  # Derived feature (redundant with app_usage_min)
	"swipe_right_label",     # Derived feature (redundant with swipe_ratio)
	"zodiac_sign",           # Irrelevant to our problem statement
]

# NOTE: the education keys use a curly apostrophe (’, U+2019) to match the dataset.
INCOME_MAP = {
	"Very Low": 1,
	"Low": 2,
	"Lower-Middle": 3,
	"Middle": 4,
	"Upper-Middle": 5,
	"High": 6,
	"Very High": 7,
}

EDUCATION_MAP = {
	"No Formal Education": 1,
	"High School": 2,
	"Diploma": 3,
	"Associate’s": 3,  # Mapped to the same level as Diploma
	"Bachelor’s": 4,
	"Master’s": 5,
	"MBA": 5,          # Mapped to the same level as Master's
	"PhD": 6,
	"Postdoc": 7,
}

CONTINUOUS_COLS = [
	"app_usage_min", "swipe_ratio", "likes_received", "mutual_matches",
	"profile_pics_count", "bio_length", "message_sent_count", "emoji_usage_rate",
	"height_cm", "weight_kg", "age", "last_active_hour",
]

PERSONA_FEATURES = [
	"income_bracket", "education_level", "app_usage_min",
	"swipe_ratio", "message_sent_count", "likes_received",
]

COLS_TO_ENCODE = [
	"gender", "sexual_orientation", "location_type",
	"swipe_time_of_day", "relationship_intent", "body_type",
]

ENGAGEMENT_FEATURES = ["app_usage_min", "swipe_ratio", "message_sent_count"]

# Dropped before training: text targets, leakage columns, and the raw engagement
# features now represented by the PCA components.
LEAKAGE_AND_REPLACED_COLS = [
	"is_high_tier",
	"is_high_tier_encoded",
	"match_outcome",
	"match_outcome_encoded",
	"mutual_matches",
	"app_usage_min",
	"swipe_ratio",
	"message_sent_count",
]

CLASS_NAMES = ["Low/Average", "High Tier"]
CHOSEN_MODEL = "Gradient Boosting" 


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
	"""Notebook cells 6-38: clean, encode, scale, cluster and PCA-reduce the data."""
	df = pd.read_csv(csv_path)

	df = df.rename(columns=COLUMN_RENAMES)
	df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns])

	df["income_bracket"] = df["income_bracket"].map(INCOME_MAP)
	df["education_level"] = df["education_level"].map(EDUCATION_MAP)

	# Intersect with the columns actually present so the pipeline tolerates the base
	# dataset (which lacks age/height_cm/weight_kg/body_type/relationship_intent).
	scale_cols = [col for col in CONTINUOUS_COLS if col in df.columns]
	scaler = StandardScaler()
	df[scale_cols] = scaler.fit_transform(df[scale_cols])

	median_matches = df["mutual_matches"].median()
	df["is_high_tier"] = df["mutual_matches"].apply(
		lambda x: "High Tier (1)" if x > median_matches else "Low/Average (0)"
	)
	df["is_high_tier_encoded"] = df["mutual_matches"].apply(
		lambda x: 1 if x > median_matches else 0
	)

	persona_cols = [col for col in PERSONA_FEATURES if col in df.columns]
	kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
	df["progression_cluster"] = kmeans.fit_predict(df[persona_cols])

	encode_cols = [col for col in COLS_TO_ENCODE if col in df.columns]
	df_final = pd.get_dummies(df, columns=encode_cols, drop_first=True)

	pca = PCA(n_components=2, random_state=42)
	pca_results = pca.fit_transform(df_final[ENGAGEMENT_FEATURES])
	df_final["engagement_pca_1"] = pca_results[:, 0]
	df_final["engagement_pca_2"] = pca_results[:, 1]

	return df_final


def _split(df_final: pd.DataFrame):
	"""Notebook cell 41: stratified 80/20 split after dropping leakage columns."""
	safe_to_drop = [col for col in LEAKAGE_AND_REPLACED_COLS if col in df_final.columns]
	y = df_final["is_high_tier_encoded"]
	X = df_final.drop(columns=safe_to_drop)
	return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _model_lineup() -> dict[str, Any]:
	return {
		"Logistic Regression": LogisticRegression(C=100, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42),
		"Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
		"SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        
		# 👇 YOUR EXACT WINNING PARAMETERS GO HERE
		"Gradient Boosting": GradientBoostingClassifier(
            learning_rate=0.01, 
            max_depth=3, 
            n_estimators=200, 
            random_state=42
        ),
        
		"Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
	}


def _positive_scores(model: Any, X: pd.DataFrame) -> Any:
	"""Positive-class scores for the ROC curve. SVC has no predict_proba (probability=True
	is slow), so fall back to its decision_function."""
	if hasattr(model, "predict_proba"):
		return model.predict_proba(X)[:, 1]
	return model.decision_function(X)


def _evaluate_model(model: Any, X_test: pd.DataFrame, y_test: Any) -> dict[str, Any]:
	"""Full per-model diagnostics: headline metrics + classification report, confusion
	matrix and ROC curve."""
	predictions = model.predict(X_test)
	report = classification_report(
		y_test, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
	)
	report_df = (
		pd.DataFrame(report)
		.transpose()
		.drop(index="accuracy", errors="ignore")
		.reset_index()
		.rename(columns={"index": "Class"})
	)
	fpr, tpr, _ = roc_curve(y_test, _positive_scores(model, X_test))
	return {
		"accuracy": float(accuracy_score(y_test, predictions)),
		"precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
		"recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
		"f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
		"report": report_df,
		"confusion": confusion_matrix(y_test, predictions).tolist(),
		"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
	}


def _train_and_evaluate_impl(csv_path: str) -> dict[str, Any]:
	df_final = load_and_preprocess(csv_path)
	X_train, X_test, y_train, y_test = _split(df_final)

	# Train each model once; keep full per-model diagnostics (Model tab) and a leaderboard
	# of headline metrics (Compare tab).
	lineup = _model_lineup()
	models: dict[str, Any] = {}
	leaderboard: list[dict[str, Any]] = []
	for name, model in lineup.items():
		model.fit(X_train, y_train)
		evaluation = _evaluate_model(model, X_test, y_test)
		models[name] = evaluation
		leaderboard.append({
			"Model": name,
			"Accuracy": evaluation["accuracy"],
			"Precision": evaluation["precision"],
			"Recall": evaluation["recall"],
			"F1": evaluation["f1"],
		})
	leaderboard.sort(key=lambda row: row["Accuracy"], reverse=True)

	return {
		"models": models,
		"model_names": list(lineup.keys()),
		"leaderboard": leaderboard,
		"chosen": CHOSEN_MODEL,
		"n_train": int(X_train.shape[0]),
		"n_features": int(X_train.shape[1]),
		"class_names": CLASS_NAMES,
	}


@st.cache_data(show_spinner="Training models on the dataset (first load only)…")
def train_and_evaluate(csv_path: str, mtime: float) -> dict[str, Any]:
	"""Cached entry point. ``mtime`` is part of the cache key so editing/replacing the
	CSV retrains; it is otherwise unused."""
	return _train_and_evaluate_impl(csv_path)
