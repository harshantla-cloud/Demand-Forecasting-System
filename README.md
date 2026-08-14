# 📊 Demand Forecasting App

An end-to-end machine learning project that predicts product demand from pricing, promotion, and inventory signals — trained on 76,000 retail transaction records and deployed as an interactive **Streamlit** web app.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🖼️ App Preview

**Input Panel** — the user enters product and pricing details:

![App Input Screen](Project%20Image/App%20Dashboard.jpeg)

**Prediction Output** — the model returns a demand estimate instantly:

![App Prediction Screen](Project%20Image/Prediction%20by%20model.jpeg)

---

## 📌 Project Overview

Retailers routinely over-order or under-order stock because demand is hard to predict manually. This project builds a supervised regression model that estimates **unit demand** for a product based on:

- Selling price and discount
- Ongoing promotions
- Competitor pricing
- Current inventory level
- Product category

The trained model is wrapped in a simple Streamlit UI, so a non-technical user (e.g. a store manager) can enter these values and get an instant demand forecast — no code required.

---

## 🗂️ Dataset

- **Size:** ~76,000 rows of daily store-level sales records
- **Raw columns:** `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Units Ordered`, `Price`, `Discount`, `Weather Condition`, `Promotion`, `Competitor Pricing`, `Seasonality`, `Epidemic`, `Demand`
- **Target variable:** `Demand`

### Feature Engineering (EDA notebook)
During exploration (`analysis.ipynb`), the following features were engineered to understand demand patterns:
- `Year`, `Month`, `Day`, `Weekday` — extracted from `Date`
- `Discounted Price` = `Price × (1 − Discount / 100)`
- `Sell Through Rate` = `Units Sold / Inventory Level`

These informed the exploratory analysis (seasonality trends, category-level demand, promotion impact, etc.) and produced the cleaned dataset `preprocessed_demand_forcasting_data.csv`.

### Final Model Features
For the production model, six features were selected based on relevance and simplicity for real-time input:

| Feature | Type |
|---|---|
| Price | Numeric |
| Discount | Numeric |
| Inventory Level | Numeric |
| Promotion | Binary (0/1) |
| Competitor Pricing | Numeric |
| Category | Categorical (Label Encoded) |

---

## 🔍 Exploratory Data Analysis

Key insights from `analysis.ipynb`:
- Demand distribution and outliers were reviewed via histograms and boxplots.
- Demand was compared across `Category`, `Weather Condition`, and `Seasonality`.
- Promotion periods showed a noticeably higher average demand.
- Daily and monthly demand trends were plotted to check for seasonality.
- Relationship between `Discounted Price` and `Demand` was visualized to confirm pricing sensitivity.

---

## 🤖 Model Building

**Algorithm:** `XGBRegressor` (XGBoost)

**Approach:**
1. Categorical feature (`Category`) encoded using `LabelEncoder`.
2. Data split 80/20 into train and test sets.
3. Hyperparameters tuned using `RandomizedSearchCV` — 25 candidate combinations, 3-fold cross-validation, optimized for **Mean Absolute Error**.
4. Best-performing model retrained on the full training set.

**Best hyperparameters found:**

```
n_estimators: 200
max_depth: 6
learning_rate: 0.1
subsample: 1.0
colsample_bytree: 0.7
min_child_weight: 1
```

### Feature Importance

| Feature | Importance |
|---|---|
| Promotion | 47.8% |
| Category | 29.4% |
| Price | 9.9% |
| Competitor Pricing | 6.0% |
| Discount | 4.7% |
| Inventory Level | 2.1% |

**Promotion** and **Category** are by far the strongest drivers of demand in this dataset — actionable insight for inventory and marketing planning.

The final model and encoder are serialized with `pickle`:
- `xgboost_demand_model.pkl`
- `label_encoder.pkl`

---

## 🖥️ Web App (Streamlit)

`app.py` loads the trained model and encoder, then renders a dark-themed dashboard where a user can:

1. Enter `Price`, `Discount`, `Inventory Level`
2. Select `Promotion`, `Competitor Price`, and `Category`
3. Click **Predict Demand** to get an instant unit forecast

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Deployment / UI | Streamlit |
| Model Persistence | Pickle |

---

## 📁 Project Structure

```
Demand Forcasting Model/
├── app.py                                   # Streamlit web app
├── analysis.ipynb                           # EDA & feature engineering
├── machine_learning.ipynb                   # Model training & tuning
├── demand_forecasting.csv                   # Raw dataset
├── preprocessed_demand_forcasting_data.csv  # Cleaned/engineered dataset
├── xgboost_demand_model.pkl                 # Trained XGBoost model
├── label_encoder.pkl                        # Fitted label encoder
├── background.png                           # App background image
├── Project Image/                           # Screenshots & diagrams
└── README.md
```

---

## 🚀 Running the App Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd "Demand Forcasting Model"

# 2. Install dependencies
pip install streamlit pandas scikit-learn xgboost

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📈 Future Improvements

- Add time-series features (lag values, rolling averages) to capture trend/seasonality directly in the model.
- Track model performance with MAE, RMSE, and R² on a held-out validation set and log it here.
- Add batch prediction (CSV upload) to the Streamlit app.
- Deploy the app to Streamlit Community Cloud for a live public demo link.

---

## 👤 Author

Feel free to connect if you'd like to discuss this project or explore collaboration opportunities.
