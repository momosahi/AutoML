import pickle
import shap
import streamlit as st
import pandas as pd


def create_model(df, model_type):
    """perform modeling on the data using pycaret.

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to be used for modeling
    model_type : pycaret module
        the pycaret module to be used for modeling
    """
    if df is None:
        st.error("Error: 'df' is not defined.")
    else:
        target = st.selectbox("Select your Target", df.columns)
        model_type.setup(df, target=target)
        setup_df = model_type.pull()
        if "setup_df" in locals():
            st.info("ML Experiment setup")
            st.dataframe(setup_df)
        else:
            st.warning("No data available for display")
        best_model = model_type.compare_models()
        compare_df = model_type.pull()
        if "compare_df" in locals():
            st.info("ML Model generated")
            st.dataframe(compare_df)
        else:
            st.warning("No model to display.")
        if best_model is not None:
            try:
                model_type.save_model(best_model, "output/best_model")
            except Exception as e:
                st.error(f"Error saving the model: {str(e)}")


def model_explanation():
    """Explains the best model generated by the ML task."""
    with open("output/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("input/source_data.csv")
    target = st.selectbox("Select your Target", df.columns)
    X = df.drop(target, axis=1)

    # Get feature importance
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    st.write("Feature Importance:")
    st.dataframe(feature_importance)

    # Use SHAP to explain the model's predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar"))


def download_model():
    """Downloads the best model generated by the ML task."""
    with open("output/best_model.pkl", "rb") as f:
        st.download_button("Download Model", f, file_name="output/best_model.pkl")
