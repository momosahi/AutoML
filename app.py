import os

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
from pycaret import regression, classification, clustering  # type: ignore

from src.data_uploading_profiling import upload_data, data_profiling, load_data
from src.model_creation import create_model

SOURCE_DATA_PATH = "input/source_data.csv"
OUTPUT_MODEL_PATH = "output/best_model"


def handle_upload(file_path: str) -> pd.DataFrame:
    """This function handles the data uploading process.
    It allows users to upload a csv file and returns a pandas DataFrame.
    """
    return upload_data(file_path)


def handle_profiling(file_path: str = SOURCE_DATA_PATH):
    """This function handles the data profiling process.
    parameters: file_path: str
    """
    st.title("Automated Exploratory Data Analysis")
    data_profiling(file_path)


def handle_ml(input_data_path: str, output_model_path: str):
    """This function handles the machine learning process."""
    st.title("Machine Learning Algo")
    task = st.radio(
        "choose the ML task", ["Classification", "Regression", "Clustering"]
    )
    if task == "Classification":
        create_model(classification, input_data_path, output_model_path)
    elif task == "Regression":
        create_model(regression, input_data_path, output_model_path)
    elif task == "Clustering":
        create_model(clustering, input_data_path, output_model_path)


def handle_prediction():
    pass


def handle_download():
    pass


def main():
    """
    This is the main function of the AutoMLStream application.
    It allows users to navigate through different options and perform various tasks.
    """
    df: pd.DataFrame = pd.DataFrame()

    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("AutoMLStream")
        choice = st.radio(
            "Navigation", ["Upload", "Profiling", "ML", "Prediction", "Download"]
        )
        st.info(
            "This application allows you to build automated Machine Learning pipeline using streamlit web app"
        )

    if choice == "Upload":
        df = handle_upload(SOURCE_DATA_PATH)

    if choice == "Profiling":
        st.title("Automated Exploratory Data Analysis")
        handle_profiling()

    if choice == "ML":
        handle_ml(SOURCE_DATA_PATH, OUTPUT_MODEL_PATH)

    if choice == "Prediction":
        handle_prediction()

    if choice == "Download":
        handle_download()

    return df


if __name__ == "__main__":
    if os.path.exists(SOURCE_DATA_PATH):
        try:
            df = load_data(SOURCE_DATA_PATH)
        except FileNotFoundError as e:
            st.error(f"Error reading csv file: {str(e)}")
    main()
