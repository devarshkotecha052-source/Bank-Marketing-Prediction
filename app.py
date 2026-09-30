import pandas as pd
import joblib
import streamlit as st


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1 {
        color: #0f172a;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        color: #0f172a;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 1.5rem;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #0f766e, #115e59);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 6px 18px rgba(15, 118, 110, 0.20);
    }

    .recommendation-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
        margin-bottom: 6px;
    }

    .recommendation-model {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .recommendation-text {
        font-size: 14px;
        opacity: 0.92;
    }

    .result-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
    }

    .result-small {
        font-size: 14px;
        color: #64748b;
        margin-top: 6px;
    }

    .section-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-top: 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    div.stButton > button {
        background: linear-gradient(135deg, #0f766e, #115e59);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.7rem 1.2rem;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #115e59, #134e4a);
        color: white;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

lr_model = joblib.load("model/bank_marketing_lr_model.pkl")
dt_model = joblib.load("model/bank_marketing_dt_model.pkl")
bagging_model = joblib.load("model/bank_marketing_bagging_model.pkl")
rf_model = joblib.load("model/bank_marketing_rf_model.pkl")
extra_trees_model = joblib.load("model/bank_marketing_extra_trees_model.pkl")
adaboost_model = joblib.load("model/bank_marketing_adaboost_model.pkl")
gradient_boosting_model = joblib.load("model/bank_marketing_gradient_boosting_model.pkl")
xgb_model = joblib.load("model/bank_marketing_xgboost_model.pkl")

lr_tuned_model = joblib.load("model/bank_marketing_lr_tuned_model.pkl")
adaboost_tuned_model = joblib.load("model/bank_marketing_adaboost_tuned_model.pkl")
gradient_boosting_tuned_model = joblib.load("model/bank_marketing_gradient_boosting_tuned_model.pkl")
xgb_tuned_model = joblib.load("model/bank_marketing_xgboost_tuned_model.pkl")

stacking_model = joblib.load("model/bank_marketing_stacking_model.pkl")
voting_model = joblib.load("model/bank_marketing_voting_model.pkl")

feature_columns = joblib.load("model/bank_marketing_feature_columns.pkl")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Bank Marketing Prediction")

st.markdown(
    '<div class="subtitle">Predict customer subscription likelihood using multiple machine learning models.</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------

st.markdown(
    """
    <div class="recommendation-card">
        <div class="recommendation-title">Recommended Model</div>
        <div class="recommendation-model">Soft Voting Ensemble</div>
        <div class="recommendation-text">
            Selected model from the Bank Marketing analysis.
            In Batch CSV mode, campaign results are generated using
            the model selected by the user.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Prediction Mode
# ---------------------------------------------------------

prediction_mode = st.radio(
    "Prediction Mode",
    [
        "Single Customer",
        "Batch CSV"
    ],
    horizontal=True
)


# =========================================================
# SINGLE CUSTOMER MODE
# =========================================================

if prediction_mode == "Single Customer":

    # -----------------------------------------------------
    # Customer Information
    # -----------------------------------------------------

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

    with col2:
        balance = st.number_input(
            "Balance",
            value=1000.0
        )

    with col3:
        day = st.number_input(
            "Day of Contact",
            min_value=1,
            max_value=31,
            value=15
        )


    col4, col5, col6 = st.columns(3)

    with col4:
        campaign = st.number_input(
            "Contacts During Current Campaign",
            min_value=1,
            value=2
        )

    with col5:
        pdays = st.number_input(
            "Days Since Previous Contact",
            value=-1
        )

    with col6:
        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            value=0
        )


    # -----------------------------------------------------
    # Categorical Information
    # -----------------------------------------------------

    st.subheader("Customer and Campaign Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ]
        )

    with col2:
        marital = st.selectbox(
            "Marital Status",
            [
                "married",
                "single",
                "divorced"
            ]
        )

    with col3:
        education = st.selectbox(
            "Education",
            [
                "primary",
                "secondary",
                "tertiary",
                "unknown"
            ]
        )


    col4, col5, col6 = st.columns(3)

    with col4:
        default = st.selectbox(
            "Credit in Default",
            [
                "no",
                "yes"
            ]
        )

    with col5:
        housing = st.selectbox(
            "Housing Loan",
            [
                "no",
                "yes"
            ]
        )

    with col6:
        loan = st.selectbox(
            "Personal Loan",
            [
                "no",
                "yes"
            ]
        )


    col7, col8, col9 = st.columns(3)

    with col7:
        contact = st.selectbox(
            "Contact Communication Type",
            [
                "cellular",
                "telephone",
                "unknown"
            ]
        )

    with col8:
        month = st.selectbox(
            "Month",
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec"
            ]
        )

    with col9:
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            [
                "failure",
                "success",
                "other",
                "unknown"
            ]
        )


    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------

    previously_contacted = int(pdays != -1)

    campaign_intensity = pd.cut(
        pd.Series([campaign]),
        bins=[0, 1, 3, 5, 10, float("inf")],
        labels=["1", "2-3", "4-5", "6-10", "10+"]
    ).iloc[0]

    previous_exposure = pd.cut(
        pd.Series([previous]),
        bins=[-1, 0, 2, 5, 10, float("inf")],
        labels=["0", "1-2", "3-5", "6-10", "10+"]
    ).iloc[0]

    age_group = pd.cut(
        pd.Series([age]),
        bins=[0, 25, 35, 45, 55, 65, float("inf")],
        labels=["<25", "25-35", "36-45", "46-55", "56-65", "65+"]
    ).iloc[0]

    previous_campaign_experience = (
        str(previous_exposure)
        + " | "
        + poutcome
    )


    # -----------------------------------------------------
    # Create Input DataFrame
    # -----------------------------------------------------

    input_df = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome],
        "previously_contacted": [previously_contacted],
        "campaign_intensity": [campaign_intensity],
        "previous_exposure": [previous_exposure],
        "age_group": [age_group],
        "previous_campaign_experience": [previous_campaign_experience]
    })

    input_df = input_df[feature_columns]


    # -----------------------------------------------------
    # Prediction Button
    # -----------------------------------------------------

    if st.button("Predict Subscription"):

        # -------------------------------------------------
        # All 14 Models
        # -------------------------------------------------

        y_pred_lr = lr_model.predict(input_df)[0]
        y_prob_lr = lr_model.predict_proba(input_df)[0]

        y_pred_dt = dt_model.predict(input_df)[0]
        y_prob_dt = dt_model.predict_proba(input_df)[0]

        y_pred_bagging = bagging_model.predict(input_df)[0]
        y_prob_bagging = bagging_model.predict_proba(input_df)[0]

        y_pred_rf = rf_model.predict(input_df)[0]
        y_prob_rf = rf_model.predict_proba(input_df)[0]

        y_pred_extra_trees = extra_trees_model.predict(input_df)[0]
        y_prob_extra_trees = extra_trees_model.predict_proba(input_df)[0]

        y_pred_adaboost = adaboost_model.predict(input_df)[0]
        y_prob_adaboost = adaboost_model.predict_proba(input_df)[0]

        y_pred_gradient_boosting = gradient_boosting_model.predict(input_df)[0]
        y_prob_gradient_boosting = gradient_boosting_model.predict_proba(input_df)[0]

        y_pred_xgb = xgb_model.predict(input_df)[0]
        y_prob_xgb = xgb_model.predict_proba(input_df)[0]

        y_pred_lr_tuned = lr_tuned_model.predict(input_df)[0]
        y_prob_lr_tuned = lr_tuned_model.predict_proba(input_df)[0]

        y_pred_adaboost_tuned = adaboost_tuned_model.predict(input_df)[0]
        y_prob_adaboost_tuned = adaboost_tuned_model.predict_proba(input_df)[0]

        y_pred_gradient_boosting_tuned = gradient_boosting_tuned_model.predict(input_df)[0]
        y_prob_gradient_boosting_tuned = gradient_boosting_tuned_model.predict_proba(input_df)[0]

        y_pred_xgb_tuned = xgb_tuned_model.predict(input_df)[0]
        y_prob_xgb_tuned = xgb_tuned_model.predict_proba(input_df)[0]

        y_pred_stacking = stacking_model.predict(input_df)[0]
        y_prob_stacking = stacking_model.predict_proba(input_df)[0]

        y_pred_voting = voting_model.predict(input_df)[0]
        y_prob_voting = voting_model.predict_proba(input_df)[0]


        # -------------------------------------------------
        # Recommended Model Result
        # -------------------------------------------------

        recommended_prediction = (
            "Yes" if y_pred_voting == 1 else "No"
        )

        recommended_yes_probability = y_prob_voting[1] * 100
        recommended_no_probability = y_prob_voting[0] * 100


        st.subheader("Recommended Model Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.markdown(
                """
                <div class="result-card">
                    <div class="result-title">Recommended Model</div>
                    <div class="result-value">Soft Voting Ensemble</div>
                    <div class="result-small">Project selected model</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Prediction</div>
                    <div class="result-value">{recommended_prediction}</div>
                    <div class="result-small">Term Deposit Subscription</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Yes Probability</div>
                    <div class="result-value">{recommended_yes_probability:.2f}%</div>
                    <div class="result-small">Estimated probability of subscription</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.progress(
            int(round(recommended_yes_probability))
        )

        st.caption(
            f"Yes: {recommended_yes_probability:.2f}%  |  "
            f"No: {recommended_no_probability:.2f}%"
        )


        # -------------------------------------------------
        # Model Results Table
        # -------------------------------------------------

        results_df = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Bagging Classifier",
                "Random Forest",
                "Extra Trees",
                "AdaBoost",
                "Gradient Boosting",
                "XGBoost",
                "Tuned Logistic Regression",
                "Tuned AdaBoost",
                "Tuned Gradient Boosting",
                "Tuned XGBoost",
                "Stacking Ensemble",
                "Soft Voting Ensemble"
            ],
            "Prediction": [
                "Yes" if y_pred_lr == 1 else "No",
                "Yes" if y_pred_dt == 1 else "No",
                "Yes" if y_pred_bagging == 1 else "No",
                "Yes" if y_pred_rf == 1 else "No",
                "Yes" if y_pred_extra_trees == 1 else "No",
                "Yes" if y_pred_adaboost == 1 else "No",
                "Yes" if y_pred_gradient_boosting == 1 else "No",
                "Yes" if y_pred_xgb == 1 else "No",
                "Yes" if y_pred_lr_tuned == 1 else "No",
                "Yes" if y_pred_adaboost_tuned == 1 else "No",
                "Yes" if y_pred_gradient_boosting_tuned == 1 else "No",
                "Yes" if y_pred_xgb_tuned == 1 else "No",
                "Yes" if y_pred_stacking == 1 else "No",
                "Yes" if y_pred_voting == 1 else "No"
            ],
            "Yes Probability (%)": [
                round(y_prob_lr[1] * 100, 2),
                round(y_prob_dt[1] * 100, 2),
                round(y_prob_bagging[1] * 100, 2),
                round(y_prob_rf[1] * 100, 2),
                round(y_prob_extra_trees[1] * 100, 2),
                round(y_prob_adaboost[1] * 100, 2),
                round(y_prob_gradient_boosting[1] * 100, 2),
                round(y_prob_xgb[1] * 100, 2),
                round(y_prob_lr_tuned[1] * 100, 2),
                round(y_prob_adaboost_tuned[1] * 100, 2),
                round(y_prob_gradient_boosting_tuned[1] * 100, 2),
                round(y_prob_xgb_tuned[1] * 100, 2),
                round(y_prob_stacking[1] * 100, 2),
                round(y_prob_voting[1] * 100, 2)
            ],
            "No Probability (%)": [
                round(y_prob_lr[0] * 100, 2),
                round(y_prob_dt[0] * 100, 2),
                round(y_prob_bagging[0] * 100, 2),
                round(y_prob_rf[0] * 100, 2),
                round(y_prob_extra_trees[0] * 100, 2),
                round(y_prob_adaboost[0] * 100, 2),
                round(y_prob_gradient_boosting[0] * 100, 2),
                round(y_prob_xgb[0] * 100, 2),
                round(y_prob_lr_tuned[0] * 100, 2),
                round(y_prob_adaboost_tuned[0] * 100, 2),
                round(y_prob_gradient_boosting_tuned[0] * 100, 2),
                round(y_prob_xgb_tuned[0] * 100, 2),
                round(y_prob_stacking[0] * 100, 2),
                round(y_prob_voting[0] * 100, 2)
            ]
        })


        # -------------------------------------------------
        # Dashboard
        # -------------------------------------------------

        st.subheader("Prediction Dashboard")

        yes_model_count = (
            results_df["Prediction"] == "Yes"
        ).sum()

        no_model_count = (
            results_df["Prediction"] == "No"
        ).sum()

        average_yes_probability = (
            results_df["Yes Probability (%)"].mean()
        )


        dashboard_col1, dashboard_col2, dashboard_col3 = st.columns(3)

        with dashboard_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Models Predicting Yes</div>
                    <div class="result-value">{yes_model_count}</div>
                    <div class="result-small">out of 14 models</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dashboard_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Models Predicting No</div>
                    <div class="result-value">{no_model_count}</div>
                    <div class="result-small">out of 14 models</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dashboard_col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">Average Yes Probability</div>
                    <div class="result-value">{average_yes_probability:.2f}%</div>
                    <div class="result-small">across all models</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # Graph 1 - Prediction Distribution
        # -------------------------------------------------

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.write("### Prediction Distribution")

        single_prediction_chart = (
            results_df["Prediction"]
            .value_counts()
            .reindex(
                ["Yes", "No"],
                fill_value=0
            )
        )

        st.bar_chart(
            single_prediction_chart
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # Graph 2 - Model Probability Comparison
        # -------------------------------------------------

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.write("### Yes Probability by Model")

        single_probability_chart = (
            results_df
            .set_index("Model")[
                ["Yes Probability (%)"]
            ]
        )

        st.bar_chart(
            single_probability_chart
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # Detailed Model Results
        # -------------------------------------------------

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.write("### Model Comparison")

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# BATCH CSV MODE
# =========================================================

if prediction_mode == "Batch CSV":

    # -----------------------------------------------------
    # Model Selection
    # -----------------------------------------------------

    st.subheader("Campaign Prediction")

    selected_model_name = st.selectbox(
        "Select Prediction Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Bagging Classifier",
            "Random Forest",
            "Extra Trees",
            "AdaBoost",
            "Gradient Boosting",
            "XGBoost",
            "Tuned Logistic Regression",
            "Tuned AdaBoost",
            "Tuned Gradient Boosting",
            "Tuned XGBoost",
            "Stacking Ensemble",
            "Soft Voting Ensemble"
        ]
    )


    selected_model = voting_model

    if selected_model_name == "Logistic Regression":
        selected_model = lr_model

    if selected_model_name == "Decision Tree":
        selected_model = dt_model

    if selected_model_name == "Bagging Classifier":
        selected_model = bagging_model

    if selected_model_name == "Random Forest":
        selected_model = rf_model

    if selected_model_name == "Extra Trees":
        selected_model = extra_trees_model

    if selected_model_name == "AdaBoost":
        selected_model = adaboost_model

    if selected_model_name == "Gradient Boosting":
        selected_model = gradient_boosting_model

    if selected_model_name == "XGBoost":
        selected_model = xgb_model

    if selected_model_name == "Tuned Logistic Regression":
        selected_model = lr_tuned_model

    if selected_model_name == "Tuned AdaBoost":
        selected_model = adaboost_tuned_model

    if selected_model_name == "Tuned Gradient Boosting":
        selected_model = gradient_boosting_tuned_model

    if selected_model_name == "Tuned XGBoost":
        selected_model = xgb_tuned_model

    if selected_model_name == "Stacking Ensemble":
        selected_model = stacking_model

    if selected_model_name == "Soft Voting Ensemble":
        selected_model = voting_model


    # -----------------------------------------------------
    # CSV Upload
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        batch_df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python"
        )

        st.write(
            f"Customers uploaded: **{len(batch_df):,}**"
        )


        # -------------------------------------------------
        # Required Columns
        # -------------------------------------------------

        required_batch_columns = [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "campaign",
            "pdays",
            "previous",
            "poutcome"
        ]

        missing_columns = list(
            set(required_batch_columns)
            - set(batch_df.columns)
        )


        if len(missing_columns) > 0:

            st.error(
                "The uploaded CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )

        else:

            # ---------------------------------------------
            # Remove Target / Unused Column if Present
            # ---------------------------------------------

            if "y" in batch_df.columns:
                batch_df = batch_df.drop(
                    columns=["y"]
                )

            if "duration" in batch_df.columns:
                batch_df = batch_df.drop(
                    columns=["duration"]
                )


            # ---------------------------------------------
            # Batch Feature Engineering
            # ---------------------------------------------

            batch_df["previously_contacted"] = (
                batch_df["pdays"] != -1
            ).astype(int)


            batch_df["campaign_intensity"] = pd.cut(
                batch_df["campaign"],
                bins=[0, 1, 3, 5, 10, float("inf")],
                labels=[
                    "1",
                    "2-3",
                    "4-5",
                    "6-10",
                    "10+"
                ]
            )


            batch_df["previous_exposure"] = pd.cut(
                batch_df["previous"],
                bins=[-1, 0, 2, 5, 10, float("inf")],
                labels=[
                    "0",
                    "1-2",
                    "3-5",
                    "6-10",
                    "10+"
                ]
            )


            batch_df["age_group"] = pd.cut(
                batch_df["age"],
                bins=[0, 25, 35, 45, 55, 65, float("inf")],
                labels=[
                    "<25",
                    "25-35",
                    "36-45",
                    "46-55",
                    "56-65",
                    "65+"
                ]
            )


            batch_df["previous_campaign_experience"] = (
                batch_df["previous_exposure"].astype(str)
                + " | "
                + batch_df["poutcome"].astype(str)
            )


            # ---------------------------------------------
            # Prepare Model Input
            # ---------------------------------------------

            batch_input_df = batch_df[feature_columns]


            # ---------------------------------------------
            # Prediction Button
            # ---------------------------------------------

            if st.button("Predict Customers"):

                batch_predictions = (
                    selected_model.predict(
                        batch_input_df
                    )
                )

                batch_probabilities = (
                    selected_model.predict_proba(
                        batch_input_df
                    )
                )


                # -----------------------------------------
                # Prediction Output
                # -----------------------------------------

                batch_output_df = batch_df.copy()

                batch_output_df.insert(
                    0,
                    "Customer Number",
                    range(
                        1,
                        len(batch_output_df) + 1
                    )
                )

                batch_output_df["Selected Model"] = (
                    selected_model_name
                )

                batch_output_df["Prediction"] = [
                    "Yes" if prediction == 1 else "No"
                    for prediction in batch_predictions
                ]

                batch_output_df["Yes Probability (%)"] = (
                    batch_probabilities[:, 1] * 100
                ).round(2)

                batch_output_df["No Probability (%)"] = (
                    batch_probabilities[:, 0] * 100
                ).round(2)


                # -----------------------------------------
                # Campaign Summary
                # -----------------------------------------

                customers_processed = len(
                    batch_output_df
                )

                predicted_yes = (
                    batch_predictions == 1
                ).sum()

                predicted_no = (
                    batch_predictions == 0
                ).sum()

                yes_rate = (
                    predicted_yes
                    / customers_processed
                    * 100
                )

                average_yes_probability = (
                    batch_output_df[
                        "Yes Probability (%)"
                    ].mean()
                )


                st.subheader("Campaign Dashboard")


                summary_col1, summary_col2, summary_col3, summary_col4 = (
                    st.columns(4)
                )


                with summary_col1:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title">Selected Model</div>
                            <div class="result-value">{selected_model_name}</div>
                            <div class="result-small">Used for campaign prediction</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with summary_col2:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title">Customers Processed</div>
                            <div class="result-value">{customers_processed:,}</div>
                            <div class="result-small">Uploaded customers</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with summary_col3:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title">Predicted Yes</div>
                            <div class="result-value">{predicted_yes:,}</div>
                            <div class="result-small">{yes_rate:.2f}% of customers</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                with summary_col4:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title">Average Yes Probability</div>
                            <div class="result-value">{average_yes_probability:.2f}%</div>
                            <div class="result-small">Across uploaded customers</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # -----------------------------------------
                # Recommendation
                # -----------------------------------------

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <div class="recommendation-title">
                            Campaign Recommendation
                        </div>

                        <div class="recommendation-model">
                            Based on {selected_model_name}
                        </div>

                        <div class="recommendation-text">
                            The selected model predicts
                            {predicted_yes:,} out of {customers_processed:,}
                            customers as potential term-deposit subscribers.
                            Customers with higher predicted Yes probability
                            can be prioritised for campaign outreach.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # -----------------------------------------
                # Graph 1 - Prediction Distribution
                # -----------------------------------------

                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )

                st.write("### Prediction Distribution")

                batch_prediction_chart = (
                    batch_output_df["Prediction"]
                    .value_counts()
                    .reindex(
                        ["Yes", "No"],
                        fill_value=0
                    )
                )

                st.bar_chart(
                    batch_prediction_chart
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # -----------------------------------------
                # Graph 2 - Probability Distribution
                # -----------------------------------------

                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )

                st.write("### Yes Probability Distribution")

                probability_labels = [
                    "0-10%",
                    "10-20%",
                    "20-30%",
                    "30-40%",
                    "40-50%",
                    "50-60%",
                    "60-70%",
                    "70-80%",
                    "80-90%",
                    "90-100%"
                ]

                probability_bins = pd.cut(
                    batch_output_df[
                        "Yes Probability (%)"
                    ],
                    bins=[
                        0,
                        10,
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        100
                    ],
                    labels=probability_labels,
                    include_lowest=True
                )

                probability_distribution = (
                    probability_bins
                    .value_counts()
                    .reindex(
                        probability_labels,
                        fill_value=0
                    )
                )

                st.bar_chart(
                    probability_distribution
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # -----------------------------------------
                # Top Customers
                # -----------------------------------------

                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )

                st.write(
                    "### Highest-Priority Customers by Yes Probability"
                )

                top_customers = (
                    batch_output_df[
                        [
                            "Customer Number",
                            "Prediction",
                            "Yes Probability (%)",
                            "No Probability (%)"
                        ]
                    ]
                    .sort_values(
                        "Yes Probability (%)",
                        ascending=False
                    )
                    .head(10)
                )

                st.dataframe(
                    top_customers,
                    hide_index=True,
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # -----------------------------------------
                # Full Results
                # -----------------------------------------

                st.markdown(
                    '<div class="section-card">',
                    unsafe_allow_html=True
                )

                st.write("### Customer Prediction Results")

                st.dataframe(
                    batch_output_df,
                    hide_index=True,
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # -----------------------------------------
                # Download Results
                # -----------------------------------------

                download_data = batch_output_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=download_data,
                    file_name="bank_marketing_predictions.csv",
                    mime="text/csv"
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown(
    '<div class="footer">Bank Marketing Machine Learning Application</div>',
    unsafe_allow_html=True
)