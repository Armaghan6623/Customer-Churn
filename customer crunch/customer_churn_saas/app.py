import os
import sys

# Ensure local package import works when running `python app.py` / streamlit run
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from customer_churn_saas.model import ChurnSaaSModel


def run():
    st.set_page_config(page_title="Customer Churn SaaS", layout="wide")
    st.title("📊 Customer Churn Analytics Ecosystem")

    model = ChurnSaaSModel()

    tab1, tab2 = st.tabs(["Risk Profiling", "Macro Forecasting"])

    with tab1:
        st.header("Individual Risk Profiler")
        credit_score = st.slider("Credit Score", 350, 850, 600, step=1)
        geography = st.selectbox("Geographic Cluster", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender Designation", ["Male", "Female"])
        age = st.slider("Customer Age (Years)", 18, 92, 40, step=1)

        tenure = st.slider("Tenure Window (Years with Bank)", 0, 10, 5, step=1)
        balance = st.number_input(
            "Liquid Account Balance ($)",
            min_value=0.0,
            max_value=250000.0,
            value=75000.0,
            step=500.0,
        )
        num_products = st.slider("Active Financial Products", 1, 4, 2, step=1)

        has_credit_card = st.radio("Holds Credit Card?", ["Yes", "No"])
        is_active_member = st.radio(
            "Maintains Active Engagement Status?", ["Yes", "No"]
        )
        estimated_salary = st.number_input(
            "Estimated Annual Income ($)",
            min_value=0.0,
            max_value=200000.0,
            value=100000.0,
            step=1000.0,
        )

        payload = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_credit_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary,
        }

        if st.button("Run Prediction"):
            with st.spinner("Running inference..."):
                res = model.predict_customer_churn_probability(payload)

            st.metric("Churn probability", f"{res['churn_probability'] * 100:.2f}%")
            if res["prediction"] == 1:
                st.error("High churn risk")
            else:
                st.success("Low churn risk")

    with tab2:
        st.header("Macro-Horizon Forecasting")
        if st.button("Generate 6-month Forecast"):
            with st.spinner("Generating forecast plot..."):
                out = model.generate_future_forecast(steps=6)

            st.subheader("SARIMAX Forecast")
            st.dataframe(out["sarima"], use_container_width=True)

            st.subheader("XGBoost Forecast")
            st.dataframe(out["xgb"], use_container_width=True)

            st.image(
                out["plot_path"],
                caption="SARIMAX vs XGBoost forecasts",
                use_container_width=True,
            )


if __name__ == "__main__":
    run()
