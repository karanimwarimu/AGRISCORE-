st.set_page_config(page_title="AgriScore", layout="wide")
st.title("🌾 AgriScore: Alternative Credit Scoring for Smallholder Farmers")
st.markdown("**Kenya & Tanzania** | Machine Learning + Explainable AI")

explainer = load_explainer()

page = st.sidebar.radio("Select Mode", 
    ["Single Farmer Scoring", "Batch Scoring (CSV)", "Global Insights"])

# ===================== SINGLE FARMER SCORING =====================
if page == "Single Farmer Scoring":
    st.header("Single Farmer Credit Scoring")

    with st.form("farmer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hh_size = st.number_input("Household Size", 1, 30, 5)
            hh_dependency_ratio = st.number_input("HH Dependency Ratio", 0.0, 5.0, 1.2)
            age_manager = st.number_input("Age of Manager", 18, 90, 42)
            farm_size = st.number_input("Farm Size (ha)", 0.1, 20.0, 2.5)
            livestock = st.number_input("Livestock Count", 0, 50, 8)
            dist_market = st.number_input("Distance to Market (km)", 0, 100, 12)

        with col2:
            yield_per_ha = st.number_input("Yield per ha (kg)", 100, 5000, 740)
            balance_stability = st.slider("Balance Stability", 0.0, 1.0, 0.81)
            climate_risk_score = st.slider("Climate Risk Score", 0.0, 1.0, 0.38)
            tx_frequency = st.number_input("Transaction Frequency", 0, 100, 28)
            input_purchase_ratio = st.slider("Input Purchase Ratio", 0.0, 1.0, 0.68)
            cashflow_volatility = st.slider("Cashflow Volatility", 0.0, 1.0, 0.22)

        with col3:
            country = st.selectbox("Country", ["Kenya", "Tanzania"])
            main_crop_clean = st.selectbox("Main Crop", ["maize", "beans", "rice", "coffee", "other"])
            soil_npk = st.number_input("Soil NPK", 0, 100, 45)
            drought_risk = st.slider("Drought Risk", 0.0, 1.0, 0.35)
            crop_health_index = st.slider("Crop Health Index", 0.0, 1.0, 0.78)
            inorganic_fertilizer = st.number_input("Inorganic Fertilizer (kg)", 0, 500, 120)

        submitted = st.form_submit_button("Score This Farmer", type="primary")

        if submitted:
            farmer_data = {
                'hh_size': hh_size,
                'hh_dependency_ratio': hh_dependency_ratio,
                'age_manager': age_manager,
                'farm_size': farm_size,
                'livestock': livestock,
                'dist_market': dist_market,
                'yield_per_ha': yield_per_ha,
                'balance_stability': balance_stability,
                'climate_risk_score': climate_risk_score,
                'tx_frequency': tx_frequency,
                'input_purchase_ratio': input_purchase_ratio,
                'cashflow_volatility': cashflow_volatility,
                'country': country,
                'main_crop_clean': main_crop_clean,
                'soil_npk': soil_npk,
                'drought_risk': drought_risk,
                'crop_health_index': crop_health_index,
                'inorganic_fertilizer': inorganic_fertilizer,
                # Add remaining fields with defaults
                'hh_primary_education': 1,
                'hh_formal_education': 8,
                'farm_work': 180,
                'fertilizer_per_ha': 48,
                'yield_kg_ha': 1850,
                'yield_stability': 0.75,
                'rainfall_mm': 850,
                'soil_quality_index': 65,
                'climate_stress_index': 0.42,
                'total_volume': 145000,
                'avg_transaction': 5200,
                'fraud_rate': 0.08,
                'input_efficiency': 1.45,
                'hh_shock': 0,
                'drought_shock': 1,
            }

            with st.spinner("Calculating Credit Score..."):
                result = score_farmer(farmer_data)

            # Display Results
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Credit Score", result['credit_score'])
            col_b.metric("Risk Tier", result['risk_tier'])
            col_c.metric("Default Probability", f"{result['pd_probability']:.1%}")
            col_d.metric("Recommendation", result['recommendation'])

            st.info(f"**Explanation**: {result['explanation_text']}")

            # SHAP Force Plot
            st.subheader("SHAP Explanation (Force Plot)")
            X = preprocess_for_scoring(pd.DataFrame([farmer_data]))
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap.force_plot(explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
                           shap_values[0], X.iloc[0], matplotlib=True, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

# ===================== BATCH SCORING =====================
elif page == "Batch Scoring (CSV)":
    st.header("Batch Scoring")
    uploaded = st.file_uploader("Upload CSV file with farmer data", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)}** farmers")
        if st.button("Score All Farmers"):
            with st.spinner("Scoring batch..."):
                results = score_farmer(df)
                result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)
            st.download_button("Download Results", result_df.to_csv(index=False), "agrisore_results.csv")

# ===================== GLOBAL INSIGHTS =====================
else:
    st.header("Global Model Insights")
    if st.button("Generate Global SHAP Summary Plot"):
        with st.spinner("Generating..."):
            df_bg = pd.read_csv("Final_dataset.csv").sample(800, random_state=42)
            X_bg = preprocess_for_scoring(df_bg)
            plot_path = plot_shap_summary(explainer, X_bg)
            st.image(plot_path, use_column_width=True)

st.caption("AgriScore v0.1 | Capstone Project - Group 3")


# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from agriscore.scoring import score_farmer
from agriscore.explain import load_explainer, plot_shap_summary
from agriscore.preprocessing import preprocess_for_scoring