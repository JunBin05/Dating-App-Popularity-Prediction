from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model_pipeline


APP_TITLE = "Dating Profile Popularity Dashboard"
APP_SUBTITLE = (
	"Interactive demo for predicting whether a profile is likely to land in the "
	"high-tier mutual-match group."
)

DATA_SEED = 42
DEFAULT_ROWS = 600

# Source dataset for the notebook's trained models (shown in the Model tab). The other
# tabs use a synthetic demo dataset; this one trains on the real CSV when present.
RAW_DATASET_PATH = Path("dating_app_behavior_dataset.csv")

GENDER_OPTIONS = ["Woman", "Man", "Non-binary", "Prefer not to say"]
ORIENTATION_OPTIONS = ["Straight", "Gay", "Lesbian", "Bisexual", "Pansexual", "Asexual"]
LOCATION_OPTIONS = ["Urban", "Suburban", "Rural", "Remote"]
INCOME_OPTIONS = [
	"Low",
	"Lower-Middle",
	"Middle",
	"Upper-Middle",
	"High",
	"Very High",
	"Luxury",
]
EDUCATION_OPTIONS = [
	"No Formal Education",
	"High School",
	"Associate Degree",
	"Bachelor's Degree",
	"Master's Degree",
	"MBA",
	"PhD",
]
RELATIONSHIP_OPTIONS = ["Long-term", "Short-term", "Casual", "Friendship", "Open to all"]
BODY_TYPE_OPTIONS = ["Slim", "Average", "Athletic", "Curvy", "Plus-size"]
SWIPE_TIME_OPTIONS = ["Morning", "Afternoon", "Evening", "Late Night"]
INTEREST_OPTIONS = [
	"Fitness",
	"Travel",
	"Music",
	"Food",
	"Tech",
	"Art",
	"Gaming",
	"Fashion",
	"Books",
	"Sports",
	"Parenting",
	"Languages",
]

NUMERIC_FEATURES = [
	"age",
	"height_cm", 
	"weight_kg",
	"app_usage_min",
	"swipe_ratio",
	"message_sent_count",
	"last_active_hour",
	"emoji_usage_rate",
	"profile_pics_count",
	"bio_length",
	"response_time_hours",
	"likes_received",
	"mutual_matches",
]

PREDICTION_FEATURES = [
	"age",
	"height_cm", 
	"weight_kg",
	"app_usage_min",
	"swipe_ratio",
	"message_sent_count",
	"last_active_hour",
	"emoji_usage_rate",
	"profile_pics_count",
	"bio_length",
	"response_time_hours",
	"gender",
	"sexual_orientation",
	"location_type",
	"income_bracket",
	"education_level",
	"relationship_intent",
	"body_type",
	"swipe_time_of_day",
]

FEATURE_RANGES = {
	"age": (18, 60),
	"height_cm": (140, 220), 
	"weight_kg": (40, 150), 
	"app_usage_min": (5, 360),
	"swipe_ratio": (0.0, 1.0),
	"message_sent_count": (0, 80),
	"last_active_hour": (0, 23),
	"emoji_usage_rate": (0.0, 1.0),
	"profile_pics_count": (1, 12),
	"bio_length": (20, 600),
	"response_time_hours": (0.0, 72.0),
	"likes_received": (0, 400),
	"mutual_matches": (0, 120),
}


@dataclass(frozen=True)
class ModelProfile:
	name: str
	description: str
	weights: dict[str, float]
	category_bonuses: dict[str, dict[str, float]]
	bias: float


MODEL_PROFILES = [
	ModelProfile(
		name="Balanced",
		description="Mixes engagement, profile quality, and consistency.",
		weights={
			"age": -0.05,
			"app_usage_min": 0.55,
			"swipe_ratio": 1.25,
			"message_sent_count": 0.35,
			"last_active_hour": 0.10,
			"emoji_usage_rate": 0.20,
			"profile_pics_count": 0.45,
			"bio_length": 0.35,
			"response_time_hours": -0.75,
		},
		category_bonuses={
			"income_bracket": {"High": 0.35, "Very High": 0.5, "Luxury": 0.65},
			"education_level": {"Master's Degree": 0.2, "MBA": 0.25, "PhD": 0.3},
			"relationship_intent": {"Long-term": 0.35, "Friendship": 0.1},
			"body_type": {"Athletic": 0.15, "Average": 0.08},
			"location_type": {"Urban": 0.2, "Suburban": 0.1},
			"swipe_time_of_day": {"Evening": 0.12, "Late Night": 0.08},
		},
		bias=-0.25,
	),
	ModelProfile(
		name="Engagement First",
		description="Rewards fast, active, and highly responsive users.",
		weights={
			"age": -0.02,
			"app_usage_min": 0.75,
			"swipe_ratio": 1.05,
			"message_sent_count": 0.6,
			"last_active_hour": 0.18,
			"emoji_usage_rate": 0.45,
			"profile_pics_count": 0.22,
			"bio_length": 0.15,
			"response_time_hours": -1.0,
		},
		category_bonuses={
			"relationship_intent": {"Casual": 0.18, "Short-term": 0.1},
			"swipe_time_of_day": {"Late Night": 0.2, "Evening": 0.1},
			"location_type": {"Urban": 0.15},
		},
		bias=-0.1,
	),
	ModelProfile(
		name="Profile Quality",
		description="Leans on completeness and profile polish signals.",
		weights={
			"age": 0.01,
			"app_usage_min": 0.25,
			"swipe_ratio": 0.85,
			"message_sent_count": 0.2,
			"last_active_hour": 0.08,
			"emoji_usage_rate": 0.1,
			"profile_pics_count": 0.7,
			"bio_length": 0.6,
			"response_time_hours": -0.55,
		},
		category_bonuses={
			"education_level": {"Bachelor's Degree": 0.12, "Master's Degree": 0.2, "MBA": 0.25, "PhD": 0.3},
			"income_bracket": {"Upper-Middle": 0.08, "High": 0.18, "Very High": 0.25},
			"body_type": {"Athletic": 0.12, "Curvy": 0.05, "Average": 0.05},
			"relationship_intent": {"Long-term": 0.22},
		},
		bias=-0.2,
	),
]


def clamp(value: float, low: float, high: float) -> float:
	return float(max(low, min(high, value)))


def sigmoid(value: float) -> float:
	return float(1.0 / (1.0 + np.exp(-value)))


@st.cache_data(show_spinner=False)
def build_demo_dataset(rows: int = DEFAULT_ROWS) -> pd.DataFrame:
	rng = np.random.default_rng(DATA_SEED)

	df = pd.DataFrame(
		{
			"age": rng.integers(18, 60, size=rows),
			"app_usage_min": rng.integers(10, 360, size=rows),
			"swipe_ratio": np.round(rng.beta(2.2, 2.0, size=rows), 2),
			"message_sent_count": rng.poisson(18, size=rows),
			"last_active_hour": rng.integers(0, 24, size=rows),
			"emoji_usage_rate": np.round(rng.beta(2.5, 2.5, size=rows), 2),
			"profile_pics_count": rng.integers(1, 13, size=rows),
			"bio_length": rng.integers(25, 600, size=rows),
			"response_time_hours": np.round(rng.gamma(2.0, 6.0, size=rows), 2),
			"gender": rng.choice(GENDER_OPTIONS, size=rows),
			"sexual_orientation": rng.choice(ORIENTATION_OPTIONS, size=rows),
			"location_type": rng.choice(LOCATION_OPTIONS, size=rows, p=[0.5, 0.28, 0.14, 0.08]),
			"income_bracket": rng.choice(INCOME_OPTIONS, size=rows),
			"education_level": rng.choice(EDUCATION_OPTIONS, size=rows),
			"relationship_intent": rng.choice(RELATIONSHIP_OPTIONS, size=rows),
			"body_type": rng.choice(BODY_TYPE_OPTIONS, size=rows),
			"swipe_time_of_day": rng.choice(SWIPE_TIME_OPTIONS, size=rows),
		}
	)

	interest_counts = rng.integers(2, 6, size=rows)
	df["interest_tags"] = [", ".join(sorted(rng.choice(INTEREST_OPTIONS, size=count, replace=False))) for count in interest_counts]

	score = (
		0.015 * df["app_usage_min"]
		+ 1.65 * df["swipe_ratio"]
		+ 0.12 * df["message_sent_count"]
		- 0.04 * df["response_time_hours"]
		+ 0.16 * df["profile_pics_count"]
		+ 0.002 * df["bio_length"]
		+ 0.03 * df["last_active_hour"]
		+ 0.3 * df["income_bracket"].isin(["High", "Very High", "Luxury"]).astype(float)
		+ 0.18 * df["education_level"].isin(["Master's Degree", "MBA", "PhD"]).astype(float)
		+ 0.1 * df["location_type"].eq("Urban").astype(float)
	)
	df["likes_received"] = np.maximum(0, np.round(score * 18 + rng.normal(15, 5, size=rows))).astype(int)
	df["mutual_matches"] = np.maximum(0, np.round(score * 4 + rng.normal(10, 4, size=rows))).astype(int)

	median_matches = df["mutual_matches"].median()
	df["is_high_tier"] = np.where(df["mutual_matches"] > median_matches, "High Tier (1)", "Low/Average (0)")
	df["is_high_tier_encoded"] = (df["mutual_matches"] > median_matches).astype(int)
	return df


def normalize_numeric(value: float, feature_name: str) -> float:
	low, high = FEATURE_RANGES[feature_name]
	if high == low:
		return 0.0
	centered = (clamp(value, low, high) - low) / (high - low)
	return centered * 2.0 - 1.0


def score_interests(selected_interests: list[str]) -> float:
	if not selected_interests:
		return -0.1
	diversity = min(len(set(selected_interests)) / 6.0, 1.0)
	theme_bonus = 0.04 if {"Travel", "Fitness", "Music"}.intersection(selected_interests) else 0.0
	return 0.08 * len(selected_interests) + 0.2 * diversity + theme_bonus


def score_profile(profile: dict[str, Any], model: ModelProfile) -> tuple[float, list[tuple[str, float]]]:
	contributions: list[tuple[str, float]] = []
	raw_score = model.bias

	for feature_name, weight in model.weights.items():
		value = float(profile[feature_name])
		contribution = weight * normalize_numeric(value, feature_name)
		raw_score += contribution
		contributions.append((feature_name, contribution))

	for feature_name, bonus_map in model.category_bonuses.items():
		selected_value = str(profile[feature_name])
		contribution = bonus_map.get(selected_value, 0.0)
		raw_score += contribution
		contributions.append((feature_name, contribution))

	interest_bonus = score_interests(profile.get("interest_tags", []))
	raw_score += interest_bonus
	contributions.append(("interest_tags", interest_bonus))

	probability = sigmoid(raw_score)
	return probability, contributions


def build_profile_from_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
	return {
		"age": int(inputs["age"]),
		"height_cm": float(inputs["height_cm"]),
		"weight_kg": float(inputs["weight_kg"]),
		"likes_received": int(inputs["likes_received"]),
		"app_usage_min": float(inputs["app_usage_min"]),
		"swipe_ratio": float(inputs["swipe_ratio"]),
		"message_sent_count": float(inputs["message_sent_count"]),
		"last_active_hour": float(inputs["last_active_hour"]),
		"emoji_usage_rate": float(inputs["emoji_usage_rate"]),
		"profile_pics_count": float(inputs["profile_pics_count"]),
		"bio_length": float(inputs["bio_length"]),
		"response_time_hours": float(inputs["response_time_hours"]),
		"gender": inputs["gender"],
		"sexual_orientation": inputs["sexual_orientation"],
		"location_type": inputs["location_type"],
		"income_bracket": inputs["income_bracket"],
		"education_level": inputs["education_level"],
		"relationship_intent": inputs["relationship_intent"],
		"body_type": inputs["body_type"],
		"swipe_time_of_day": inputs["swipe_time_of_day"],
	}


def render_kpi_cards(df: pd.DataFrame) -> None:
	total_profiles = len(df)
	high_tier_share = df["is_high_tier_encoded"].mean() * 100 if "is_high_tier_encoded" in df.columns else 0.0
	median_matches = df["mutual_matches"].median() if "mutual_matches" in df.columns else 0
	if "response_time_hours" in df.columns:
		fourth_metric = ("Avg response hours", f"{df['response_time_hours'].mean():.1f}")
	elif "app_usage_min" in df.columns:
		fourth_metric = ("Avg app usage (min)", f"{df['app_usage_min'].mean():.0f}")
	else:
		fourth_metric = ("Avg messages sent", f"{df['message_sent_count'].mean():.0f}")

	card_cols = st.columns(4)
	metrics = [
		("Profiles", f"{total_profiles:,}"),
		("High-tier share", f"{high_tier_share:.1f}%"),
		("Median mutual matches", f"{median_matches:.0f}"),
		fourth_metric,
	]
	for column, (label, value) in zip(card_cols, metrics, strict=False):
		with column:
			st.metric(label, value)


def render_dataset_overview(df: pd.DataFrame) -> None:
	overview_left, overview_right = st.columns([1, 1])

	with overview_left:
		target_chart = px.bar(
			df["is_high_tier"].value_counts().reset_index(),
			x="is_high_tier",
			y="count",
			title="Target distribution",
			color="is_high_tier",
			color_discrete_sequence=["#0f766e", "#f59e0b"],
		)
		target_chart.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
		st.plotly_chart(target_chart, use_container_width=True)

	with overview_right:
		present_numeric = [feature for feature in NUMERIC_FEATURES if feature in df.columns]
		numeric_summary = df[present_numeric].describe().T.reset_index().rename(columns={"index": "feature"})
		st.dataframe(numeric_summary, use_container_width=True, hide_index=True)


def render_explore_tab(df: pd.DataFrame) -> None:
	st.subheader("Explore the dataset")

	left, right = st.columns([1, 1])

	with left:
		category_options = [
			column
			for column in ["gender", "location_type", "income_bracket", "education_level", "relationship_intent"]
			if column in df.columns
		]
		selected_category = st.selectbox("Categorical view", category_options)
		category_counts = df[selected_category].value_counts().reset_index()
		category_counts.columns = [selected_category, "count"]
		chart = px.bar(category_counts, x=selected_category, y="count", title=f"Distribution of {selected_category}")
		chart.update_layout(margin=dict(l=0, r=0, t=50, b=0))
		st.plotly_chart(chart, use_container_width=True)

	with right:
		numeric_options = [
			column
			for column in ["app_usage_min", "swipe_ratio", "message_sent_count", "profile_pics_count", "bio_length", "response_time_hours"]
			if column in df.columns
		]
		selected_numeric = st.selectbox("Numeric view", numeric_options)
		box = px.box(df, x="is_high_tier", y=selected_numeric, points="outliers", color="is_high_tier")
		box.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
		st.plotly_chart(box, use_container_width=True)

	st.markdown("### Correlation heatmap")
	corr = df[[feature for feature in NUMERIC_FEATURES if feature in df.columns]].corr(numeric_only=True)
	heatmap = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", aspect="auto")
	heatmap.update_layout(margin=dict(l=0, r=0, t=50, b=0))
	st.plotly_chart(heatmap, use_container_width=True)


def render_prediction_tab() -> None:
	st.subheader("Predict a profile")
	st.caption("The default implementation uses scoring profiles that mirror the notebook's feature logic. Saved model artifacts can be wired in next.")

	with st.form("prediction_form"):
		st.caption("Hover over the (?) icon next to each field for details on what to enter.")
		form_left, form_right = st.columns(2)
		
		with form_left:
			age = st.slider("Age", 18, 60, 29)
			height_cm = st.slider("Height (cm)", 140, 220, 170, help="User's height in centimeters")
			weight_kg = st.slider("Weight (kg)", 40, 150, 65, help="User's weight in kilograms")   
			app_usage_min = st.slider("Daily app usage (minutes)", 5, 360, 120, 
				help="Total time spent active on the app per day.")
			swipe_ratio = st.slider("Swipe-right ratio", 0.0, 1.0, 0.54, 0.01, 
				help="0.0 means you swipe left on everyone (super picky). 1.0 means you swipe right on everyone.")
			message_sent_count = st.slider("Messages sent", 0, 80, 18, 
				help="Total number of outgoing messages sent by the user.")
			likes_received = st.slider("Likes received", 0, 400, 45, 
				help="Current number of likes the profile has received.")
			last_active_hour = st.slider("Last active hour", 0, 23, 20, 
				help="The hour of the day the user usually logs off (e.g., 20 = 8:00 PM).")
			response_time_hours = st.slider("Average response time (hours)", 0.0, 72.0, 8.0, 0.5, 
				help="How many hours it takes to reply to a match. Lower is faster.")
				
		with form_right:
			emoji_usage_rate = st.slider("Emoji usage rate", 0.0, 1.0, 0.38, 0.01, 
				help="0.0 means text-only. 1.0 means every single message has emojis.")
			profile_pics_count = st.slider("Profile pictures", 1, 12, 5, 
				help="How many distinct photos are uploaded to the profile (e.g., 5 photos).")
			bio_length = st.slider("Bio length (characters)", 20, 600, 180, 
				help="Character count of the text bio. 20 is one short sentence; 600 is a mini-essay.")
			gender = st.selectbox("Gender", GENDER_OPTIONS)
			sexual_orientation = st.selectbox("Sexual orientation", ORIENTATION_OPTIONS)
			location_type = st.selectbox("Location type", LOCATION_OPTIONS)
			income_bracket = st.selectbox("Income bracket", INCOME_OPTIONS)
			education_level = st.selectbox("Education level", EDUCATION_OPTIONS)
			relationship_intent = st.selectbox("Relationship intent", RELATIONSHIP_OPTIONS)
			body_type = st.selectbox("Body type", BODY_TYPE_OPTIONS)
			swipe_time_of_day = st.selectbox("Most active time", SWIPE_TIME_OPTIONS, 
				help="The time of day when you do the most swiping.")

		submitted = st.form_submit_button("Generate prediction")

	if not submitted:
		st.info("Adjust the profile details, then generate a prediction.")
		return

	profile = build_profile_from_inputs(
		{
			"age": age,
			"height_cm": height_cm,
			"weight_kg": weight_kg, 
			"likes_received": likes_received,
			"app_usage_min": app_usage_min,
			"swipe_ratio": swipe_ratio,
			"message_sent_count": message_sent_count,
			"last_active_hour": last_active_hour,
			"emoji_usage_rate": emoji_usage_rate,
			"profile_pics_count": profile_pics_count,
			"bio_length": bio_length,
			"response_time_hours": response_time_hours,
			"gender": gender,
			"sexual_orientation": sexual_orientation,
			"location_type": location_type,
			"income_bracket": income_bracket,
			"education_level": education_level,
			"relationship_intent": relationship_intent,
			"body_type": body_type,
			"swipe_time_of_day": swipe_time_of_day,
		}
	)
	results = []
	for model in MODEL_PROFILES:
		probability, contributions = score_profile(profile, model)
		predicted_label = "High Tier" if probability >= 0.5 else "Low/Average"
		results.append(
			{
				"Model": model.name,
				"Probability": probability,
				"Prediction": predicted_label,
				"Top signal": max(contributions, key=lambda item: abs(item[1]))[0],
			}
		)

	result_df = pd.DataFrame(results)

	primary = result_df.iloc[0]
	top_color = "#0f766e" if primary["Prediction"] == "High Tier" else "#b45309"
	st.markdown(
		f"""
		<div style='padding: 1rem 1.2rem; border-radius: 1rem; background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(245,158,11,0.12)); border: 1px solid rgba(148,163,184,0.25);'>
			<div style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.08em; color: #475569;'>Primary result</div>
			<div style='font-size: 2rem; font-weight: 700; color: {top_color};'>{primary['Prediction']}</div>
			<div style='font-size: 1rem; color: #334155;'>Estimated high-tier probability: <strong>{primary['Probability']:.1%}</strong></div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	score_chart = px.bar(
		result_df.sort_values("Probability", ascending=True),
		x="Probability",
		y="Model",
		color="Prediction",
		orientation="h",
		color_discrete_map={"High Tier": "#0f766e", "Low/Average": "#b45309"},
		title="Model comparison",
	)
	score_chart.update_layout(margin=dict(l=0, r=0, t=50, b=0), xaxis_tickformat=".0%")
	st.plotly_chart(score_chart, use_container_width=True)

	explanation_model = MODEL_PROFILES[0]
	_, contributions = score_profile(profile, explanation_model)
	explanation = (
		pd.DataFrame(contributions, columns=["feature", "contribution"])
		.assign(abs_contribution=lambda frame: frame["contribution"].abs())
		.sort_values("abs_contribution", ascending=False)
		.head(6)
	)
	st.markdown("### Why this result")
	st.dataframe(explanation[["feature", "contribution"]], use_container_width=True, hide_index=True)


def render_compare_tab() -> None:
	st.subheader("Compare model behavior")
	st.caption("How the five trained models compare on the held-out test set.")

	if not RAW_DATASET_PATH.exists():
		st.info(
			f"Add the project dataset at `{RAW_DATASET_PATH.as_posix()}` to compare the trained models."
		)
		return

	results = model_pipeline.train_and_evaluate(
		str(RAW_DATASET_PATH), RAW_DATASET_PATH.stat().st_mtime
	)
	metrics_df = pd.DataFrame(results["leaderboard"])[["Model", "Accuracy", "Precision", "Recall", "F1"]]

	long_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
	chart = px.bar(
		long_df,
		x="Model",
		y="Score",
		color="Metric",
		barmode="group",
		title="Trained model comparison (test set)",
	)
	chart.update_layout(margin=dict(l=0, r=0, t=50, b=0), yaxis_tickformat=".0%", legend_title_text="")
	st.plotly_chart(chart, use_container_width=True)

	st.caption(
		f"Project's chosen model: **{results['chosen']}**. All five cluster near 50% "
		"(macro-averaged) — the dataset is effectively random."
	)
	st.dataframe(metrics_df.round(3), use_container_width=True, hide_index=True)


def render_batch_tab(df: pd.DataFrame) -> None:
	st.subheader("Batch scoring")
	uploaded = st.file_uploader("Upload a CSV to inspect", type=["csv"])

	if uploaded is None:
		st.info("Upload a CSV with the same columns as the demo profile schema to preview scoring.")
		st.dataframe(df.head(10), use_container_width=True, hide_index=True)
		return

	try:
		uploaded_df = pd.read_csv(uploaded)
	except Exception as exc:  # pragma: no cover - Streamlit surface
		st.error(f"Could not read file: {exc}")
		return

	missing_columns = [column for column in PREDICTION_FEATURES if column not in uploaded_df.columns]
	if missing_columns:
		st.warning("Missing columns: " + ", ".join(missing_columns))
		st.dataframe(uploaded_df.head(10), use_container_width=True, hide_index=True)
		return

	scored_rows = []
	for _, row in uploaded_df.iterrows():
		row_profile = build_profile_from_inputs({
			"age": row["age"],
			"likes_received": int(row.get("likes_received", 0)),
			"app_usage_min": row["app_usage_min"],
			"swipe_ratio": row["swipe_ratio"],
			"message_sent_count": row["message_sent_count"],
			"last_active_hour": row["last_active_hour"],
			"emoji_usage_rate": row.get("emoji_usage_rate", 0.3),
			"profile_pics_count": row["profile_pics_count"],
			"bio_length": row["bio_length"],
			"response_time_hours": row["response_time_hours"],
			"gender": row["gender"],
			"sexual_orientation": row["sexual_orientation"],
			"location_type": row["location_type"],
			"income_bracket": row["income_bracket"],
			"education_level": row["education_level"],
			"relationship_intent": row["relationship_intent"],
			"body_type": row["body_type"],
			"swipe_time_of_day": row["swipe_time_of_day"],
		})
		probability, _ = score_profile(row_profile, MODEL_PROFILES[0])
		scored_rows.append(probability)

	uploaded_df = uploaded_df.copy()
	uploaded_df["predicted_high_tier_probability"] = scored_rows
	uploaded_df["predicted_class"] = np.where(uploaded_df["predicted_high_tier_probability"] >= 0.5, "High Tier", "Low/Average")
	st.dataframe(uploaded_df.head(50), use_container_width=True, hide_index=True)


def render_model_tab() -> None:
	st.subheader("Project model & results")
	st.caption(
		"Inspect any of the five trained models. Logistic Regression is the project's chosen "
		"final model; the Compare tab weighs them all at once."
	)

	if not RAW_DATASET_PATH.exists():
		st.info(
			f"Add the project dataset at `{RAW_DATASET_PATH.as_posix()}` to populate this tab "
			"(the dating-app behaviour CSV, ~50k rows)."
		)
		return

	results = model_pipeline.train_and_evaluate(
		str(RAW_DATASET_PATH), RAW_DATASET_PATH.stat().st_mtime
	)
	chosen = results["chosen"]
	model_names = results["model_names"]

	st.markdown(
		"**Problem:** classify dating profiles into a high-performing vs. low-performing tier "
		"from demographics and in-app behaviour.  \n"
		f"**Chosen model:** {chosen} — tuned via Grid Search for optimal tree depth (3), "
        "learning rate (0.01), and iterations (200)."
	)

	selected = st.selectbox("Model to inspect", model_names, index=model_names.index(chosen))
	model = results["models"][selected]

	metric_cols = st.columns(3)
	metric_cols[0].metric("Accuracy", f"{model['accuracy']:.2%}")
	metric_cols[1].metric("Test ROC AUC", f"{model['roc']['auc']:.2f}")
	metric_cols[2].metric("Train shape", f"{results['n_train']:,} × {results['n_features']}")

	heading = f"{selected} — evaluation"
	if selected == chosen:
		heading += " (project pick)"
	st.markdown(f"### {heading}")
	st.dataframe(model["report"], use_container_width=True, hide_index=True)

	detail_left, detail_right = st.columns(2)
	with detail_left:
		class_names = results["class_names"]
		confusion_fig = px.imshow(
			model["confusion"],
			text_auto=True,
			color_continuous_scale="Blues",
			labels=dict(x="Predicted", y="Actual", color="Count"),
			x=class_names,
			y=class_names,
			title="Confusion matrix",
		)
		confusion_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), coloraxis_showscale=False)
		st.plotly_chart(confusion_fig, use_container_width=True)
	with detail_right:
		roc = model["roc"]
		roc_fig = go.Figure()
		roc_fig.add_trace(
			go.Scatter(
				x=roc["fpr"],
				y=roc["tpr"],
				mode="lines",
				name=f"ROC (AUC = {roc['auc']:.2f})",
				line=dict(color="#0f766e", width=2),
			)
		)
		roc_fig.add_trace(
			go.Scatter(
				x=[0, 1],
				y=[0, 1],
				mode="lines",
				name="Chance",
				line=dict(color="#94a3b8", width=2, dash="dash"),
			)
		)
		roc_fig.update_layout(
			title="ROC curve",
			xaxis_title="False positive rate",
			yaxis_title="True positive rate",
			margin=dict(l=0, r=0, t=50, b=0),
			legend=dict(x=0.35, y=0.05),
		)
		st.plotly_chart(roc_fig, use_container_width=True)

	st.info(
		"**Key finding:** every model lands around 50-51% accuracy (AUC ≈ 0.50) — essentially a "
		"coin flip. An AutoML sweep (PyCaret) hit the same ceiling, so in this synthetic dataset "
		"demographics and in-app behaviour do not actually determine match outcomes; the tuned "
		"Logistic Regression is already at the data's theoretical limit."
	)


def render_sidebar(df: pd.DataFrame) -> None:
	st.sidebar.title("Dashboard Controls")
	st.sidebar.write("Use the tabs to inspect the dataset or score a profile.")
	st.sidebar.caption("This implementation uses a Streamlit-native scoring demo until saved model artifacts are added.")
	st.sidebar.divider()
	st.sidebar.write("Dataset snapshot")
	st.sidebar.metric("Rows", f"{len(df):,}")
	st.sidebar.metric("Target share", f"{df['is_high_tier_encoded'].mean() * 100:.1f}%")
	st.sidebar.metric("Median matches", f"{df['mutual_matches'].median():.0f}")


def prepare_real_dataset(path: Path) -> pd.DataFrame:
	"""Load the real CSV with the column names and target the dashboard tabs expect."""
	df = pd.read_csv(path)
	df = df.rename(columns={"app_usage_time_min": "app_usage_min", "swipe_right_ratio": "swipe_ratio"})
	if "mutual_matches" in df.columns:
		median_matches = df["mutual_matches"].median()
		df["is_high_tier_encoded"] = (df["mutual_matches"] > median_matches).astype(int)
		df["is_high_tier"] = np.where(df["is_high_tier_encoded"] == 1, "High Tier (1)", "Low/Average (0)")
	return df


@st.cache_data(show_spinner=False)
def load_input_data() -> pd.DataFrame:
	if RAW_DATASET_PATH.exists():
		return prepare_real_dataset(RAW_DATASET_PATH)
	return build_demo_dataset()


def apply_page_style() -> None:
	st.markdown(
		"""
		<style>
			.stApp {
				background:
					radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 24%),
					radial-gradient(circle at top right, rgba(245,158,11,0.08), transparent 24%),
					linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
			}
			.block-container {
				padding-top: 1.5rem;
				padding-bottom: 2rem;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
	apply_page_style()

	df = load_input_data()
	if "is_high_tier_encoded" not in df.columns:
		if "mutual_matches" in df.columns:
			median_matches = df["mutual_matches"].median()
			df = df.copy()
			df["is_high_tier_encoded"] = (df["mutual_matches"] > median_matches).astype(int)
			df["is_high_tier"] = np.where(df["is_high_tier_encoded"] == 1, "High Tier (1)", "Low/Average (0)")
		else:
			df = build_demo_dataset()

	st.title(APP_TITLE)
	st.write(APP_SUBTITLE)
	render_sidebar(df)

	render_kpi_cards(df)
	st.divider()

	tab_predict, tab_explore, tab_compare, tab_batch, tab_model = st.tabs([
		"Predict",
		"Explore",
		"Compare",
		"Batch score",
		"Model",
	])

	with tab_predict:
		render_prediction_tab()

	with tab_explore:
		render_dataset_overview(df)
		render_explore_tab(df)

	with tab_compare:
		render_compare_tab()

	with tab_batch:
		render_batch_tab(df)

	with tab_model:
		render_model_tab()


if __name__ == "__main__":
	main()
