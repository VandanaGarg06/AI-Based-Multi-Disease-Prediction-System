
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Multi-Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "03_Trained_Models")
SHAP_DIR = os.path.join(BASE_DIR, "SHAP_Data")

with open(os.path.join(BASE_DIR, "input_metadata.json")) as f:
    metadata = json.load(f)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #163b65;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_shap_data(disease):
    path = os.path.join(
        SHAP_DIR,
        disease.lower().replace(" ", "_") + "_shap.npz"
    )
    data = np.load(path, allow_pickle=False)
    return data["background"], data["feature_names"]

st.title("🩺 AI-Based Multi-Disease Prediction System")

st.markdown(
    "### Intelligent Machine Learning Prediction and Explainability"
)

st.write(
    "Explore machine learning predictions for Diabetes, "
    "Heart Disease and Chronic Kidney Disease."
)

st.info(
    "Educational and research use only. This application is not "
    "a medical diagnostic tool. Predictions should not replace "
    "professional medical advice."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Disease Prediction", "About Project"]
)

if page == "Home":

    st.header("Welcome!")

    st.write(
        "This project uses machine learning algorithms to explore "
        "three disease classification tasks."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🩸 Diabetes")
        st.write("Diabetes classification model.")

    with col2:
        st.subheader("❤️ Heart Disease")
        st.write("Heart disease classification model.")

    with col3:
        st.subheader("🫘 Kidney Disease")
        st.write("CKD classification model.")

    st.divider()

    st.subheader("Project Features")

    st.markdown("""
    - Three disease prediction options
    - Machine learning classification
    - Model-estimated probability
    - SHAP-based individual explanations
    """)

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    The AI-Based Multi-Disease Prediction System is an educational
    machine learning project designed to demonstrate how classification
    algorithms and explainable AI can be used with healthcare datasets.
    """)

    st.subheader("Machine Learning Algorithms")

    st.write("""
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting
    """)

    st.subheader("Explainable AI")
    st.write(
        "SHAP is used to explain how input features contribute "
        "to individual model predictions."
    )

else:

    st.header("Disease Prediction")

    disease = st.selectbox(
        "Choose Disease",
        ["Diabetes", "Heart Disease", "CKD"]
    )

    model_file = (
        disease.lower().replace(" ", "_") + "_best_model.pkl"
    )

    model = load_model(
        os.path.join(MODEL_DIR, model_file)
    )

    st.write("### Enter Patient Information")

    feature_data = {}

    with st.form("prediction_form"):

        columns = st.columns(2)

        for i, (feature, details) in enumerate(
            metadata[disease].items()
        ):

            with columns[i % 2]:

                label = feature.replace("_", " ").title()

                if details["type"] == "categorical":

                    feature_data[feature] = st.selectbox(
                        label,
                        details["options"],
                        key=f"{disease}_{feature}"
                    )

                else:

                    feature_data[feature] = st.number_input(
                        label,
                        min_value=float(details["minimum"]),
                        max_value=float(details["maximum"]),
                        value=float(details["default"]),
                        key=f"{disease}_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame([feature_data])

        try:
            prediction = int(model.predict(input_df)[0])

            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1)

            probability = float(
                probabilities[positive_index]
            ) * 100

            st.divider()
            st.header("Prediction Results")

            if prediction == 1:
                st.error("Model prediction: Disease class")
            else:
                st.success("Model prediction: Non-disease class")

            st.metric(
                "Model-estimated disease-class probability",
                f"{probability:.2f}%"
            )

            st.caption(
                "This probability is a model output, not a clinically "
                "validated estimate of an individual's medical risk."
            )

            st.subheader("Input Summary")

            st.dataframe(
                input_df,
                use_container_width=True
            )

            st.divider()
            st.header("SHAP Explanation")

            with st.spinner("Generating explanation..."):

                background, feature_names = load_shap_data(disease)

                preprocessor = model.named_steps["preprocessing"]
                estimator = model.named_steps["model"]

                transformed_input = preprocessor.transform(input_df)

                if hasattr(transformed_input, "toarray"):
                    transformed_input = transformed_input.toarray()

                explainer = shap.Explainer(
                    estimator,
                    background,
                    feature_names=feature_names
                )

                explanation = explainer(transformed_input)

                values = explanation.values

                if values.ndim == 3:
                    explanation_to_plot = explanation[0, :, 1]
                else:
                    explanation_to_plot = explanation[0]

            st.write(
                "The following plot explains how the input features "
                "contributed to this individual prediction."
            )

            fig = plt.figure(figsize=(10, 6))

            shap.plots.waterfall(
                explanation_to_plot,
                max_display=12,
                show=False
            )

            st.pyplot(plt.gcf())
            plt.close("all")

            st.info(
                "SHAP explains the model's behaviour. It does not "
                "establish medical causation or diagnose disease."
            )

        except Exception as e:
            st.error("Prediction or SHAP error: " + str(e))
