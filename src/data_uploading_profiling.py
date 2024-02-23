"""
This module contains the functions for uploading data and performing data profiling.
@author: Sahi Gonsangbeu
@date: 2024-02-23
"""

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
from ydata_profiling import ProfileReport  # type: ignore
from streamlit_pandas_profiling import st_profile_report  # type: ignore

SOURCE_DATA_PATH = "../input/source_data.csv"


def upload_data() -> pd.DataFrame:
    """Uploads data from a csv file."""
    st.title("Upload Data")
    file = st.file_uploader("Upload a csv file", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            st.dataframe(df)
            save_data(df)
            return df
        except FileExistsError as e:
            st.error(f"Error reading csv file: {str(e)}")
    return pd.DataFrame()


def load_data(file_path: str) -> pd.DataFrame:
    """Loads the data from a csv file."""
    return pd.read_csv(file_path)


def save_data(df: pd.DataFrame) -> None:
    """Saves the dataframe to a csv file."""
    df.to_csv(SOURCE_DATA_PATH, index=False)
    st.success("Data saved successfully.")


def generate_profile_report(df: pd.DataFrame) -> ProfileReport:
    """Generates a profile report of the data."""
    with st.spinner("Generating profile report..."):
        profile_report_df = ProfileReport(df)
        return profile_report_df


def display_profile_report(profile_report_df: ProfileReport) -> None:
    """Displays the profile report of the data."""
    st_profile_report(profile_report_df)


def data_profiling(file_path: str) -> None:
    """Performs data profiling.

    Parameters:
    file_path (str): The path to the csv file.
    """
    df = load_data(file_path)
    profile_report_df = generate_profile_report(df)
    display_profile_report(profile_report_df)
