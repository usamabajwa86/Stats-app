"""
Data Handler Module
Handles data upload, validation, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO

def load_data(uploaded_file):
    """
    Load data from uploaded file
    Supports CSV, Excel (XLS, XLSX), and TXT formats
    """
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'txt':
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or TXT files.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def manual_data_entry():
    """
    Allow users to manually enter small datasets
    """
    st.subheader("Manual Data Entry")

    num_cols = st.number_input("Number of columns:", min_value=1, max_value=10, value=2)
    num_rows = st.number_input("Number of rows:", min_value=1, max_value=100, value=5)

    col_names = []
    for i in range(int(num_cols)):
        col_name = st.text_input(f"Column {i+1} name:", value=f"Column_{i+1}", key=f"col_{i}")
        col_names.append(col_name)

    st.write("Enter data (comma-separated values for each row):")
    data_dict = {col: [] for col in col_names}

    for i in range(int(num_rows)):
        row_data = st.text_input(f"Row {i+1}:", key=f"row_{i}")
        if row_data:
            values = row_data.split(',')
            for j, col in enumerate(col_names):
                if j < len(values):
                    try:
                        data_dict[col].append(float(values[j].strip()))
                    except:
                        data_dict[col].append(values[j].strip())

    if all(len(v) > 0 for v in data_dict.values()):
        df = pd.DataFrame(data_dict)
        return df
    return None

def validate_data(df):
    """
    Validate the uploaded data
    """
    if df is None or df.empty:
        st.error("Data is empty. Please upload a valid dataset.")
        return False

    # Check for appropriate data types
    st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Display data types
    with st.expander("View Data Types"):
        st.write(df.dtypes)

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        st.warning(f"Dataset contains {missing_count} missing values.")

    return True

def clean_data(df):
    """
    Provide data cleaning options
    """
    st.subheader("Data Cleaning Options")

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.write("**Missing Values Detected**")
        missing_option = st.selectbox(
            "How to handle missing values?",
            ["Keep as is", "Remove rows with missing values", "Fill with mean",
             "Fill with median", "Fill with mode", "Fill with zero"]
        )

        if missing_option == "Remove rows with missing values":
            df = df.dropna()
            st.success("Rows with missing values removed.")
        elif missing_option == "Fill with mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.success("Missing values filled with mean.")
        elif missing_option == "Fill with median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            st.success("Missing values filled with median.")
        elif missing_option == "Fill with mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            st.success("Missing values filled with mode.")
        elif missing_option == "Fill with zero":
            df = df.fillna(0)
            st.success("Missing values filled with zero.")

    # Handle outliers
    st.write("**Outlier Detection**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0:
        outlier_col = st.selectbox("Select column to check for outliers:", numeric_cols)

        if outlier_col:
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]

            if len(outliers) > 0:
                st.warning(f"Found {len(outliers)} potential outliers in {outlier_col}")
                st.write(outliers)

                remove_outliers = st.checkbox("Remove outliers?")
                if remove_outliers:
                    df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                    st.success(f"Outliers removed from {outlier_col}")

    return df

def transform_data(df):
    """
    Provide data transformation options
    """
    st.subheader("Data Transformation")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0:
        transform_col = st.selectbox("Select column to transform:", ["None"] + numeric_cols)

        if transform_col != "None":
            transform_type = st.selectbox(
                "Select transformation type:",
                ["Log transformation", "Square root transformation",
                 "Arcsine transformation", "Standardization (Z-score)"]
            )

            new_col_name = st.text_input("New column name:", value=f"{transform_col}_transformed")

            if st.button("Apply Transformation"):
                try:
                    if transform_type == "Log transformation":
                        df[new_col_name] = np.log(df[transform_col].replace(0, np.nan))
                    elif transform_type == "Square root transformation":
                        df[new_col_name] = np.sqrt(df[transform_col])
                    elif transform_type == "Arcsine transformation":
                        df[new_col_name] = np.arcsin(np.sqrt(df[transform_col]))
                    elif transform_type == "Standardization (Z-score)":
                        df[new_col_name] = (df[transform_col] - df[transform_col].mean()) / df[transform_col].std()

                    st.success(f"Transformation applied. New column: {new_col_name}")
                except Exception as e:
                    st.error(f"Error applying transformation: {str(e)}")

    return df

def get_numeric_columns(df):
    """
    Get list of numeric columns from dataframe
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """
    Get list of categorical columns from dataframe
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def prepare_grouped_data(df, group_col, value_col):
    """
    Prepare data for grouped analysis
    """
    try:
        groups = df.groupby(group_col)[value_col].apply(list).to_dict()
        return groups
    except Exception as e:
        st.error(f"Error preparing grouped data: {str(e)}")
        return None

def export_data(df, filename="cleaned_data.csv"):
    """
    Export cleaned data
    """
    csv = df.to_csv(index=False)
    return csv
