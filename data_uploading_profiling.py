import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
import ydata_profiling  # type: ignore
from streamlit_pandas_profiling import st_profile_report  # type: ignore


def upload_data():
    """upload_data: Uploads the data to be used for modeling."""
    st.title("Upload your data for modeling !")
    file = st.file_uploader("Upload your dataset here")
    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, index_col=None)
            df.to_csv("input/source_data.csv", index=None)
            st.dataframe(df)
        else:
            st.error("Please upload a CSV file.")


def data_profiling():
    """Generates a profile report of the data."""
    df = pd.read_csv("input/source_data.csv")
    if not df.empty:
        try:
            with st.spinner("Generating profile report..."):
                profile_report_df = ydata_profiling.ProfileReport(df)
            st_profile_report(profile_report_df)
        except Exception as e:
            st.error(f"Error creating profile report: {str(e)}")
    else:
        st.warning("The dataframe is empty.")
