"""
This module contains the functions to create a model using pycaret.
@author: Sahi Gonsangbeu
@date: 2024-02-23
"""

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
from data_uploading_profiling import load_data


def select_target_variable(df: pd.DataFrame):
    """Selects the target variable from the dataframe."""
    st.title("Select the target variable")
    target_variable = st.selectbox("Select the target variable", df.columns)
    return target_variable


def setup_experiment(model_type, df: pd.DataFrame, target: str):
    """Sets up the ML experiment."""
    model_type.setup(df, target=target)
    setup_df = model_type.pull()
    if not setup_df.empty:
        st.info("ML Experiment setup")
        st.dataframe(setup_df)
    else:
        st.warning("No data available for display")


def compare_models(model_type):
    """Compares the models and returns the best one."""
    best_model = model_type.compare_models()
    compare_df = model_type.pull()
    if not compare_df.empty:
        st.info("Models compared")
        st.dataframe(compare_df)
    else:
        st.warning("No data available for display")
    return best_model


def save_best_model(model_type, best_model, output_path: str):
    """Saves the best model to a file."""
    if best_model is not None:
        try:
            model_type.save_model(best_model, output_path)
            st.success("Best model saved successfully.")
        except FileNotFoundError as e:
            st.error(f"Error saving best model: {str(e)}")


def create_model(model_type, input_data_path: str, output_model_path: str):
    """Perform modeling on the data using pycaret."""
    df = load_data(input_data_path)
    if df.empty:
        return
    target = select_target_variable(df)
    setup_experiment(model_type, df, target)
    best_model = compare_models(model_type)
    save_best_model(model_type, best_model, output_model_path)
