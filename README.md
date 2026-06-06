# Dating Profile Popularity Prediction & Dashboard

An interactive machine learning project designed to predict and analyze dating profile popularity based on user behavior, demographics, and engagement metrics. This project includes a comprehensive data analysis notebook and a Streamlit-based dashboard for real-time predictions and model evaluation.

## 🚀 Overview

The goal of this project is to identify the factors that contribute to a profile being "high-tier" (based on mutual matches) and to build a predictive model that can classify profiles into popularity tiers. It explores behavioral data like app usage time, swipe ratios, and messaging habits alongside profile characteristics like education and income.

## ✨ Key Features

- **Interactive Dashboard**: A Streamlit application to explore the dataset, visualize clusters, and test the prediction model.
- **Model Bake-off**: Comparison of multiple machine learning algorithms:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Machine (SVM)
  - Multi-layer Perceptron (MLP)
- **Feature Engineering**: 
  - Implementation of **PCA (Principal Component Analysis)** for engagement metrics.
  - **K-Means Clustering** to segment users into distinct personas.
  - Mapping of ordinal features (Education, Income).
- **Data Visualization**: Rich visualizations using Plotly and Seaborn, including correlation heatmaps and feature importance plots.

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dashboard
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Running the Dashboard
Launch the interactive Streamlit app from the `dashboard/` directory:
```bash
streamlit run app.py
```

### Analysis Notebook
For the full research process, data cleaning, and initial model training, refer to the Jupyter notebook:
`WIA1006_Popularity_Prediction_Final (6).ipynb`

## 📁 Project Structure

```text
dashboard/
├── app.py                      # Main Streamlit dashboard application
├── model_pipeline.py           # ML pipeline (preprocessing, training, evaluation)
├── requirements.txt            # Python dependencies
├── dating_app_behavior_dataset.csv # The source dataset
└── WIA1006_Popularity_Prediction_Final (6).ipynb # Original research notebook
```

## 📊 Model Details

The project uses a target variable `is_high_tier` derived from the `mutual_matches` count. The pipeline handles:
- **Preprocessing**: Handling missing values, encoding categorical variables, and scaling numeric features.
- **Dimensionality Reduction**: Using PCA to condense engagement features (`app_usage_min`, `swipe_ratio`, `message_sent_count`) into a single component.
- **Evaluation**: Models are evaluated using Accuracy, Precision, Recall, and F1-score.

## 📝 Requirements

- streamlit
- pandas
- scikit-learn
- plotly
- matplotlib
- seaborn
- joblib

---
*Created as part of the Machine Learning course (WIA1006).*
