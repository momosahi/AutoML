import os
import streamlit as st
import pandas as pd
from pycaret import regression, classification, clustering

from data_uploading_profiling import upload_data, data_profiling
from model_creation import create_model, feature_importance, download_model


def main():
    """
    This is the main function of the AutoMLStream application.
    It allows users to navigate through different options and perform various tasks.
    """
    df = None

    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("AutoMLStream")
        choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Feature importance", "Download"])
        st.info("This application allows you to build automated Machine Learning pipeline using streamlit web app")

    if choice == "Upload":
        df = upload_data()

    if choice == "Profiling":
        st.title("Automated Exploratory Data Analysis")
        data_profiling()

    if choice == "ML":
        st.title("Machine Learning Algo")
        task = st.radio("choose the ML task", ["Classification", "Regression", "Clustering"])
        if task == "Classification":
            create_model(classification)

        if task == "Regression":
            create_model(regression)

        if task == "Clustering":
            create_model(clustering)
    if choice == "Feature importance":
        st.title("Feature Importance")
        feature_importance()

    if choice == "Download":
        download_model()

    return df  # Return df at the end of the function


if __name__ == "__main__":
    if os.path.exists("input/source_data.csv"):
        try:
            df = pd.read_csv("input/source_data.csv", index_col=None)
        except Exception as e:
            st.error(f"Error reading csv file: {str(e)}")
    main()
