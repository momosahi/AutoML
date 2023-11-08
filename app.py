import os
import streamlit as st
import pandas as pd

from data_uploading_profiling import upload_data, data_profiling
from model_creation import create_classification, create_regression, create_clustering, download_model


def main():
    """
    This is the main function of the AutoMLStream application.
    It allows users to navigate through different options and perform various tasks.
    """
    df = None

    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("AutoMLStream")
        choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
        st.info("This application allows you to build automated Machine Learning pipeline using streamlit web app")

    if choice == "Upload":
        df = upload_data()

    if choice == "Profiling":
        st.title("Automated Exploratory Data Analysis")
        data_profiling(df)

    if choice == "ML":
        st.title("Machine Learning Algo")
        task = st.radio("choose the ML task", ["Classification", "Regression", "Clustering"])
        if task == "Classification":
            create_classification(df)

        if task == "Regression":
            create_regression(df)

        if task == "Clustering":
            create_clustering(df)

    if choice == "Download":
        download_model()

    return df  # Return df at the end of the function


if __name__ == "__main__":
    if os.path.exists("sourceData.csv"):
        try:
            df = pd.read_csv("sourceData.csv", index_col=None)
        except Exception as e:
            st.error(f"Error reading csv file: {str(e)}")
    main()
