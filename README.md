# Predicting Economic Recessions using Leading Indicators and Network Analysis

### 💔 [View the Live Interactive Report](https://incelpenguine.github.io/Recession_Prediction_Network_Analysis/)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Usage](#setup-and-usage)
- [Conclusion](#conclusion)

---

## Project Overview

This project aims to predict U.S. economic recessions by leveraging a combination of traditional macroeconomic indicators and network analysis. The core idea is to model the economy as a complex network where indicators are nodes and their correlations are edges. By analyzing the structural properties of this network over time, we can extract novel features that may capture systemic risk and financial contagion, potentially improving the predictive power of machine learning models.

The project compares a baseline model, trained only on standard time-series features, with an enhanced model augmented by network centrality metrics. Model interpretability is explored using SHAP to understand which features are most influential in predicting economic downturns.

---

## Key Features

* **Data Acquisition**: Fetches a wide range of economic data from the Federal Reserve Economic Data (FRED) database and stock market data from Yahoo Finance.
* **Time-Series Feature Engineering**: Creates standard predictive features from raw data, including month-over-month change, year-over-year change, and 12-month rolling means and standard deviations.
* **Dynamic Network Construction**: Builds dynamic correlation-based networks for each month in the dataset.
* **Network Feature Engineering**: Calculates key centrality measures (Degree, Betweenness, Eigenvector) for each economic indicator to quantify its systemic importance over time.
* **Predictive Modeling**: Implements and compares Logistic Regression and XGBoost classification models to predict the probability of a recession within a 6-month horizon.
* **Rigorous Evaluation**: Uses a chronological train-validation-test split to prevent data leakage and evaluates models using ROC AUC, Precision, Recall, and F1-Score.
* **Model Interpretation**: Employs SHAP (SHapley Additive exPlanations) to analyze feature importance and understand the drivers behind model predictions.
* **Visualization**: Generates visualizations of the economic network at critical periods (e.g., pre-recession, during recession) and plots the evolution of network features over time.

---

## Methodology

The project is structured into a series of sequential Jupyter notebooks:

1.  **`01_data_preprocessing_and_feature_engineering.ipynb`**:
    * Loads raw data for 12 leading economic indicators (e.g., Yield Curve, Unemployment, CPI) and the S&P 500.
    * Standardizes all data to a consistent monthly frequency.
    * Engineers 48 time-series features (YoY change, rolling means, etc.).
    * Prepares the final clean dataset with features and the NBER recession indicator as the target.

2.  **`02_network_construction_and_feature_engineering.ipynb`**:
    * Constructs a monthly correlation network based on a 12-month rolling window of the base indicators.
    * Calculates three centrality measures (Degree, Betweenness, Eigenvector) for each feature in each monthly network.
    * Creates a new dataset of 144 network-based features.

3.  **`03_modeling_and_evaluation.ipynb`**:
    * Augments the traditional time-series features with the lagged network features.
    * Splits the data chronologically into training (1961-2000), validation (2001-2010), and test (2011-2024) sets.
    * Trains and evaluates two models:
        * **Baseline Model**: Logistic Regression on traditional features only.
        * **Enhanced Models**: Logistic Regression and XGBoost on the augmented (traditional + network) feature set.
    * Saves the best-performing model (`enhanced_xgb_model.joblib`) and its associated scaler.

4.  **`04_model_interpretation_with_shap.ipynb`**:
    * Loads the saved XGBoost model.
    * Performs SHAP analysis to identify the most important features in the full model.
    * Conducts an experiment by removing the top traditional features and re-training the model to see if network features become more prominent.

5.  **`05_network_visualization_and_analysis.ipynb`**:
    * Visualizes the structure of the economic network graph at critical points in time (e.g., stable periods vs. pre-recession).
    * Plots the time series of key network centrality metrics against historical recession periods to observe their behavior during economic cycles.

---

## Key Findings

* **Model Performance**: The Enhanced XGBoost model, which included network features, was the top performer. It achieved a **ROC AUC of 0.95** on the validation set, significantly outperforming the baseline model.
* **Feature Importance**: SHAP analysis revealed that the model's predictions were primarily driven by powerful, well-known macroeconomic indicators like the **OECD Leading Indicator (USALOLITONOSTSAM)**, **Yield Curve (T10Y3MM)**, and **Inflation (CPIAUCSL)**.
* **Impact of Network Features**: While the network-augmented dataset improved model performance, the network features themselves were not the primary drivers. They appeared to capture information that was largely redundant or less powerful than that contained in the standard time-series features.
* **Final Conclusion**: The project demonstrates that while network analysis provides a measurable improvement for non-linear models like XGBoost, the most critical predictors for this specific recession forecasting task remain the well-established macroeconomic variables.

---

## Technologies Used

* **Language**: Python 3.x
* **Libraries**:
    * `pandas` & `numpy` for data manipulation
    * `scikit-learn` for modeling and preprocessing
    * `xgboost` for the gradient boosting model
    * `fredapi` & `yfinance` for data acquisition
    * `networkx` for graph creation and analysis
    * `shap` for model interpretation
    * `matplotlib` & `seaborn` for visualization
    * `joblib` for saving/loading models
    * `python-dotenv` for environment variable management
* **Environment**: Jupyter Notebook

---

## Project Structure


Recession_Prediction_Network_Analysis/
│
├── data/
│   ├── raw_economic_indicators.csv
│   ├── final_prepared_data.csv
│   ├── network_features.csv
│   └── ... (other data files)
│
├── models/
│   ├── enhanced_xgb_model.joblib
│   └── scaler_enhanced.joblib
│
├── visualizations/
│   ├── shap_summary_plot_FULL.png
│   ├── network_dynamic_stable_period_2015-mid.png
│   └── ... (other plots)
│
├── 01_data_preprocessing_and_feature_engineering.ipynb
├── 02_network_construction_and_feature_engineering.ipynb
├── 03_modeling_and_evaluation.ipynb
├── 04_model_interpretation_with_shap.ipynb
├── 05_network_visualization_and_analysis.ipynb
│
├── .env.example
├── requirements.txt
└── README.md


---

## Setup and Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Recession_Prediction_Network_Analysis.git](https://github.com/your-username/Recession_Prediction_Network_Analysis.git)
    cd Recession_Prediction_Network_Analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your FRED API Key:**
    * Obtain a free API key from the [FRED website](https://fred.stlouisfed.org/docs/api/api_key.html).
    * Rename the `.env.example` file to `.env`.
    * Add your API key to the `.env` file:
        ```
        FRED_API_KEY="your_api_key_here"
        ```

5.  **Run the Jupyter Notebooks:**
    Launch Jupyter Notebook or JupyterLab and run the notebooks in sequential order, from `01` to `05`.

    ```bash
    jupyter notebook
    ```

---

## Conclusion

This project provides a comprehensive framework for recession forecasting that integrates traditional econometrics with network science. The results indicate that while network-based features can enhance a powerful non-linear model, they do not supplant the predictive dominance of established macroeconomic indicators. This suggests that for this task, the value of network analysis may lie in providing a secondary, confirmatory signal rather than being a primary predictive tool. Future work could explore more sophisticated network construction methods or different sets of economic variables.

