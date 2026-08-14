import streamlit as st
import pandas as pd
import pickle
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# DARK BACKGROUND IMAGE
# =========================================================

with open("background.png", "rb") as f:
    background_image = base64.b64encode(f.read()).decode()


st.markdown(
    f"""
    <style>

    /* ================= BACKGROUND ================= */

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


    /* ================= MAIN CONTAINER ================= */

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}


    /* ================= TITLE ================= */

    h1 {{
        color: red !important;
        font-weight: 700;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.6);
    }}


    /* ================= HEADINGS ================= */

    h2, h3 {{
        color: white !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
    }}


    /* ================= TEXT ================= */

    p {{
        color: white !important;
    }}


    /* ================= INPUT LABELS ================= */

    label {{
        color: white !important;
    }}


    /* ================= BUTTON ================= */

    .stButton > button {{
        width: 100%;
        height: 50px;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 600;
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

    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
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

    st.success(
        f"📈 Predicted Demand: {prediction} Units"
    )