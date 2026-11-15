import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="centered"
)

# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("bank_marketing_model_package_compressed.pkl")

try:
    package = load_model()
    model = package["model"]
    scaler = package["scaler"]
    feature_columns = package["feature_columns"]
    threshold = package["threshold"]
    education_order = package["education_order"]

except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# --------------------------------
# APPLICATION TITLE
# --------------------------------

st.title("🏦 Bank Marketing Predictor")

st.write(
    """
    ### Will this customer subscribe to a term deposit?

    Enter the customer's information below. 
    Our machine learning model will estimate the likelihood
    that the customer will subscribe.
    """
)

st.divider()


# --------------------------------
# CUSTOMER INPUTS
# --------------------------------

st.subheader("1. Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    job = st.selectbox(
        "Job",
        [
            "admin.", "blue-collar", "entrepreneur",
            "housemaid", "management", "retired",
            "self-employed", "services", "student",
            "technician", "unemployed", "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

    education = st.selectbox(
        "Education",
        ["primary", "secondary", "tertiary", "unknown"]
    )

with col2:
    balance = st.number_input(
        "Account Balance",
        value=1000
    )

    default = st.selectbox(
        "Has Credit in Default?",
        ["no", "yes"]
    )

    housing = st.selectbox(
        "Has Housing Loan?",
        ["no", "yes"]
    )

    loan = st.selectbox(
        "Has Personal Loan?",
        ["no", "yes"]
    )


st.divider()

st.subheader("2. Marketing Campaign Information")

col3, col4 = st.columns(2)

with col3:
    contact = st.selectbox(
        "Contact Communication Type",
        ["cellular", "telephone", "unknown"]
    )

    month = st.selectbox(
        "Last Contact Month",
        [
            "jan", "feb", "mar", "apr", "may",
            "jun", "jul", "aug", "sep", "oct",
            "nov", "dec"
        ]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

with col4:
    campaign = st.number_input(
        "Number of Contacts During This Campaign",
        min_value=1,
        value=2
    )

    previous = st.number_input(
        "Number of Contacts Before This Campaign",
        min_value=0,
        value=0
    )

    pdays = st.number_input(
        "Days Since Previous Campaign Contact (-1 = Never Contacted)",
        min_value=-1,
        value=-1
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )


st.divider()


# --------------------------------
# PREDICTION
# --------------------------------

if st.button("🔍 Predict Customer Subscription", use_container_width=True):

    # Feature engineering
    age_group = pd.cut(
        [age],
        bins=[0, 25, 35, 50, 65, 120],
        labels=[
            "under_25",
            "25_24",
            "35_49",
            "50-64",
            "65_plus"
        ]
    )[0]

    contacted_before = int(pdays != -1)

    campaign_intensity_ratio = campaign / (previous + 1)

    education_encoded = education_order[education]

    # Create customer record
    customer = pd.DataFrame([{
        "age": age,
        "balance": balance,
        "day": day,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "compaign_intensity_ratio": campaign_intensity_ratio,
        "education": education_encoded,
        "contacted_before": contacted_before,
        "default": int(default == "yes"),
        "housing": int(housing == "yes"),
        "loan": int(loan == "yes"),
        "job": job,
        "marital": marital,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "age_group": age_group
    }])

    # One-hot encode categorical features
    customer = pd.get_dummies(
        customer,
        columns=[
            "job",
            "marital",
            "contact",
            "month",
            "poutcome",
            "age_group"
        ],
        drop_first=True
    )

    # Match the exact feature columns used during training
    customer = customer.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the original scaler statistics to the
    # numeric features used by the model.
    scaler_columns = list(scaler.feature_names_in_)

    numeric_columns = package["numeric_columns"]

    for column in numeric_columns:
        if column in customer.columns and column in scaler_columns:
            index = scaler_columns.index(column)

            customer[column] = (
                customer[column] - scaler.mean_[index]
            ) / scaler.scale_[index]

    # Generate prediction probability
    probability = model.predict_proba(customer)[0][1]

    # Apply the project's 0.40 decision threshold
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("3. Prediction Results")

    st.metric(
        "Probability of Subscription",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.success(
            "✅ Likely to Subscribe: This customer is predicted "
            "to subscribe to a term deposit."
        )
    else:
        st.warning(
            "⚠️ Unlikely to Subscribe: This customer is predicted "
            "not to subscribe to a term deposit."
        )

    st.info(
        f"""
        **Decision threshold:** {threshold}

        The model predicts YES when the estimated probability
        is at least {threshold * 100:.0f}%.
        """
    )

    st.caption(
        "This is a machine learning prediction, not a guarantee "
        "of customer behaviour."
    )
