# SuperKart: Product-Store Sales Forecasting and Deployment

An end-to-end machine learning project for **SuperKart**, a retail chain of supermarkets and food marts. It predicts the total sales revenue of a product in a given store, and serves the model through a Flask API with a Streamlit front end, both packaged with Docker for Hugging Face Spaces.

## Table of Contents
- [Business Context](#business-context)
- [Objective](#objective)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models and Results](#models-and-results)
- [Deployment Architecture](#deployment-architecture)
- [Known Issues and Next Steps](#known-issues-and-next-steps)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

## Business Context
An accurate sales forecast supports decisions across an organisation. It informs regional sales strategy, territory planning and supply chain procurement, reduces forecast and pipeline risk, and gives benchmarks for assessing future trends.

## Objective
SuperKart wants to forecast the sales revenue of its outlets to optimise inventory management and regional strategy. The goal is not only a predictive model built on historical data, but a **deployed forecasting solution** that can be integrated into SuperKart's decision-making systems and used across its store network.

## Dataset
**8,763 rows × 12 columns**, with no missing values and no duplicate records.

| Feature | Description |
|---------|-------------|
| `Product_Id` | Unique product identifier (two letters followed by a number) |
| `Product_Weight` | Weight of the product |
| `Product_Sugar_Content` | Low sugar, regular or no sugar |
| `Product_Allocated_Area` | Ratio of the product's display area to the total display area in the store |
| `Product_Type` | 16 categories (meat, snack foods, dairy, canned, soft drinks, and so on) |
| `Product_MRP` | Maximum retail price |
| `Store_Id` | Unique store identifier |
| `Store_Establishment_Year` | Year the store was established (1987 to 2009) |
| `Store_Size` | High, medium or small |
| `Store_Location_City_Type` | Tier 1, 2 or 3 |
| `Store_Type` | Departmental Store, Supermarket Type 1, Supermarket Type 2 or Food Mart |
| **`Product_Store_Sales_Total`** | **Target**: total revenue of that product in that store (range 33 to 8,000) |

> The dataset file (`SuperKart.csv`) is not included in this repository.

## Project Workflow
1. **Data overview**: shape, types, duplicates and null checks. `Product_Allocated_Area` is multiplied by 100 so it reads as a percentage.
2. **Exploratory data analysis**: univariate plots (product types, numeric distributions, categorical counts), a correlation heatmap, and bivariate box plots of every numeric feature against every categorical feature.
3. **Outlier treatment**: IQR-based clipping of numeric columns.
4. **Feature selection**: `Product_Id`, `Store_Id`, `Store_Establishment_Year`, `Store_Size` and `Store_Location_City_Type` are dropped. The model uses:
   - Numeric: `Product_Weight`, `Product_Allocated_Area`, `Product_MRP`
   - Categorical: `Product_Sugar_Content`, `Product_Type`, `Store_Type`
5. **Preprocessing pipeline**: `StandardScaler` for numeric features and `OneHotEncoder` (unknown categories ignored) for categorical features, combined with a `ColumnTransformer`.
6. **Split**: 64% train, 16% validation, 20% test.
7. **Model comparison**: Bagging, Random Forest, GBM, AdaBoost, Decision Tree and XGBoost.
8. **Hyperparameter tuning** of the two best candidates (Bagging and Random Forest) with `GridSearchCV`.
9. **Serialisation** of the full pipeline with `joblib` and deployment.

## Models and Results
Random Forest and Bagging had the best validation R² in the initial comparison (about 0.94 each), so both were tuned.

Held-out **test set** results for the tuned models:

| Model | R² | RMSE | MAE | MAPE |
|-------|----|------|-----|------|
| Tuned Bagging | 0.917 | 305.7 | 172.3 | 6.0% |
| **Tuned Random Forest (selected)** | **0.926** | **289.7** | **135.8** | **4.8%** |

Tuned Random Forest settings: `n_estimators=1000`, `max_depth=8`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=0.6`, `max_samples=0.7`.

Why Random Forest was chosen:
- Highest R² on unseen test data.
- Smallest gap between train (R² 0.936) and test (R² 0.926), which indicates good generalisation.
- The untuned models fit the training data almost perfectly (R² above 0.98) but dropped on validation, so tuning with shallower trees and subsampling traded a little training fit for stability.

> RMSE and MAE are in the same units as `Product_Store_Sales_Total`. The notebook prints metrics multiplied by 100, so values shown there (for example RMSE 28,973) are 100 times the figures above.

## Deployment Architecture
```
 Streamlit front end  ──HTTP POST /v1/Sales──▶  Flask + Gunicorn back end  ──▶  joblib model pipeline
 (Docker, HF Space)                              (Docker, HF Space)
```
Three deployment folders are produced by the notebook, each with its own `app.py`, `requirements.txt` and `Dockerfile`:

| Folder | Purpose |
|--------|---------|
| `deployment_files/` | Single-container Streamlit app that loads the model directly |
| `backend_files/` | Flask REST API serving predictions at `/v1/Sales` |
| `frontend_files/` | Streamlit UI that calls the back-end API |

Live demo links (add yours):
- Front end: `https://huggingface.co/spaces/<username>/Product_Sales_Prediction_Frontend`
- Back end: `https://huggingface.co/spaces/<username>/Product_Sales_Prediction_Backend`

## Known Issues and Next Steps
These are worth fixing before relying on the deployment.

- **Back-end code needs fixing.** The Flask route in `backend_files/app.py` builds its input from undefined variables instead of the parsed JSON payload, calls `st.info()` (a Streamlit function) in the API, and still carries copy-over text from an Airbnb project ("Welcome to the Airbnb Rental Price Prediction API", "rental price", "in dollars"). It should read fields from `request.get_json()` and return the predicted sales value.
- **Front-end API URL is malformed.** It uses `https://<username>/<space>.hf.space/...`. A Hugging Face Space URL has the form `https://<username>-<space-name>.hf.space/...`.
- **Front-end inputs include `Store_Id`, `Store_Establishment_Year`, `Store_Size` and `Store_Location_City_Type`**, which the final model does not use, and the sugar-content dropdown offers a `reg` value that should be normalised to `Regular`. The `Product_Weight` input is limited to 6.6 to 19.0 while the label says 4.0 to 22.0.
- **Outlier clipping is applied before the train/test split**, which lets test-set statistics influence the clipping bounds. Fit the clipping on the training data only.
- **Treat the target as a forecast of per-product-store totals, not a time series.** The dataset has no date column, so "next quarter" forecasts depend on how the model's output is aggregated and used downstream.
- Ideas for improvement: try XGBoost and LightGBM with proper randomised search, add store attributes back and test their impact, and add cross-validated error intervals.

## Getting Started

### Prerequisites
- Python 3.9+ (the Docker images use `python:3.9-slim`)
- Docker (optional, for containerised runs)

### Install
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib streamlit flask gunicorn huggingface_hub jupyter
```

### Run the notebook
Place `SuperKart.csv` in a `data/` folder. The notebook was written for Google Colab and reads from Google Drive, so remove the `drive.mount(...)` cell and update the CSV path before running locally.

### Run the app locally
```bash
# Option 1: single-container Streamlit app
cd deployment_files
pip install -r requirements.txt
streamlit run app.py

# Option 2: separate API and UI
cd backend_files && pip install -r requirements.txt && gunicorn -w 4 -b 0.0.0.0:7860 app:Product_Sales_Predictor_api
cd frontend_files && pip install -r requirements.txt && streamlit run app.py
```

### Deploy to Hugging Face Spaces
Create a Docker Space for each of the back end and front end, then upload the folder contents with `huggingface_hub.HfApi().upload_folder(...)`. Authenticate with an access token supplied through an environment variable or Colab secret.

> **Never commit access tokens.** Keep Hugging Face tokens out of notebooks and source files. Use `os.environ["HF_TOKEN"]` or Colab's secrets instead.

## Repository Structure
```
.
├── Model_Deployment_Project.ipynb   # EDA, modelling, serialisation, deployment code
├── deployment_files/                # Streamlit app + model + Dockerfile
├── backend_files/                   # Flask API + model + Dockerfile
├── frontend_files/                  # Streamlit UI + Dockerfile
├── data/                            # (not tracked) SuperKart.csv
└── README.md
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, joblib, Flask, Gunicorn, Streamlit, Docker, Hugging Face Spaces.

## Author
<Your Name>
