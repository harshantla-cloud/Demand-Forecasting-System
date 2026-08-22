import streamlit as st
import pandas as pd
import pickle
import base64
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

BACKGROUND_PATH = BASE_DIR / "assets" / "background.png"
MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

with open(BACKGROUND_PATH, "rb") as f:
    background_image = base64.b64encode(f.read()).decode()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       APP BACKGROUND
    ===================================================== */

    .stApp {{
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.75),
                rgba(0, 0, 0, 0.75)
            ),
            url("data:image/png;base64,{background_image}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}


    /* =====================================================
       MAIN CONTAINER
    ===================================================== */

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}


    /* =====================================================
       MAIN TITLE
    ===================================================== */

    h1 {{
        color: red !important;
        font-weight: 700;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.6);
    }}


    /* =====================================================
       HEADINGS
    ===================================================== */

    h2,
    h3 {{
        color: white !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5);
    }}


    /* =====================================================
       NORMAL TEXT
    ===================================================== */

    p {{
        color: white !important;
    }}


    /* =====================================================
       INPUT LABELS
    ===================================================== */

    label {{
        color: white !important;
    }}


    /* =====================================================
       PREDICTION BUTTON
    ===================================================== */

    .stButton > button {{
        width: 100%;
        height: 52px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.35);
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
    }}


    /* =====================================================
       METRIC BOX
    ===================================================== */

    [data-testid="stMetric"] {{
        background: rgba(0, 80, 120, 0.65);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
        text-align: center;
    }}


    [data-testid="stMetricLabel"] {{
        color: white !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }}


    [data-testid="stMetricValue"] {{
        color: white !important;
        font-size: 42px !important;
        font-weight: 800 !important;
    }}


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {{
        border-color: rgba(255, 255, 255, 0.2);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL AND ENCODER
# =========================================================

@st.cache_resource
def load_artifacts():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, label_encoders = load_artifacts()


# =========================================================
# TITLE
# =========================================================

st.title("📊 Demand Forecasting App")

st.write(
    "Predict Product Demand Using Machine Learning."
)

st.divider()


# =========================================================
# INPUT FEATURES
# =========================================================

st.header("📋 Input Features")

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    price = st.number_input(
        "Price",
        min_value=0.0,
        value=50.0
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0,
        max_value=100,
        value=10
    )

    inventory_level = st.number_input(
        "Inventory Level",
        min_value=0,
        value=100
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    promotion = st.selectbox(
        "Promotion",
        [0, 1]
    )

    competitor_pricing = st.number_input(
        "Competitor Price",
        min_value=0.0,
        value=50.0
    )

    category = st.selectbox(
        "Category",
        label_encoders["Category"].classes_.tolist()
    )


st.divider()


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({

    "Price": [price],

    "Discount": [discount],

    "Inventory Level": [inventory_level],

    "Promotion": [promotion],

    "Competitor Pricing": [competitor_pricing],

    "Category": [category]

})


# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

for col, encoder in label_encoders.items():

    if col in input_data.columns:

        input_data[col] = encoder.transform(
            input_data[col]
        )


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Demand",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    prediction = int(prediction)


    # =====================================================
    # FORECAST RESULT
    # =====================================================

    st.markdown("### 📈 Forecast Result")

    result_col1, result_col2, result_col3 = st.columns(
        [1, 2, 1]
    )

    with result_col2:

        st.metric(
            label="FORECASTED DEMAND",
            value=f"{prediction} Units"
        )

        st.caption(
            "Estimated product demand based on the selected features."
        )