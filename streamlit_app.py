import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from ydata_profiling.model.alerts import AlertType

# ===================================================================
#  Step 1: Define All Your Cleaning and Handling Functions
# ===================================================================

def handle_missing(df, alert, description, log):
    """Handles columns with missing values by dropping or imputing."""
    col_name = alert.column_name
    if col_name not in df.columns:
        return # Column might have been dropped already
        
    p_missing = alert.values['p_missing']

    if p_missing > 0.6:
        df.drop(columns=[col_name], inplace=True)
        log.append(f"SUCCESS: Dropped column '{col_name}' due to >60% missing values.")
    else:
        if description.variables[col_name]['type'] == 'Numeric':
            median_val = df[col_name].median()
            df[col_name].fillna(median_val, inplace=True)
            log.append(f"SUCCESS: Imputed missing values in '{col_name}' with median ({median_val}).")
        else: 
            mode_val = df[col_name].mode()[0]
            df[col_name].fillna(mode_val, inplace=True)
            log.append(f"SUCCESS: Imputed missing values in '{col_name}' with mode ('{mode_val}').")

def handle_zeros(df, alert, log):
    """Logs information about columns with a high percentage of zeros."""
    col_name = alert.column_name
    p_zeros = alert.values['p_zeros']
    log.append(f"INFO: Column '{col_name}' has {p_zeros:.1%} zeros. No automatic action taken.")

def handle_duplicates(df, alert, log):
    """Handles duplicate rows in the dataset."""
    num_duplicates = alert.values['n_duplicates']
    df.drop_duplicates(inplace=True)
    log.append(f"SUCCESS: Removed {num_duplicates} duplicate rows.")

def handle_alphanumeric(df, col_name, log):
    """Handles feature engineering for alphanumeric columns."""
    col_name_lower = col_name.lower()

    if 'ticket' in col_name_lower or 'tkt' in col_name_lower:
        log.append(f"INFO: Processing Ticket-like column: '{col_name}'.")
        col_as_str = df[col_name].astype(str)
        df[f'{col_name}_prefix'] = col_as_str.str.replace(r'\s*\d+$', '', regex=True).str.strip()
        df[f'{col_name}_prefix'].replace('', 'NONE', inplace=True)
        df[f'{col_name}_number'] = pd.to_numeric(col_as_str.str.extract(r'(\d+)$')[0], errors='coerce')
        df.drop(columns=[col_name], inplace=True)
        log.append(f"SUCCESS: Split '{col_name}' into prefix/number and dropped original.")
        return

    elif 'cabin' in col_name_lower:
        log.append(f"INFO: Processing Cabin-like column: '{col_name}'.")
        col_as_str = df[col_name].astype(str)
        df[f'{col_name}_prefix'] = col_as_str.str[0]
        df.drop(columns=[col_name], inplace=True)
        log.append(f"SUCCESS: Created '{col_name}_prefix' and dropped original.")
        return

    else:
        log.append(f"SUCCESS: Enriched column '{col_name}' with general features.")
        col_as_str = df[col_name].astype(str)
        df[f'{col_name}_length'] = col_as_str.str.len()
        df[f'{col_name}_num_digits'] = col_as_str.str.count(r'[0-9]')

# ===================================================================
#  Step 2: Define the Main Cleaning Engine Function
# ===================================================================

def clean_and_analyze_data(df):
    cleaning_log = []
    df_original = df.copy()

    # --- Phase 1: Proactive Transformations ---
    for col_name in df.columns.copy():
        # Datetime Handling
        try:
            converted_col = pd.to_datetime(df[col_name], errors='coerce')
            non_null_percentage = converted_col.notna().sum() / len(df)
            if non_null_percentage > 0.6:
                if converted_col.dt.month.nunique() > 1 or converted_col.dt.day.nunique() > 1:
                    df[f'{col_name}_year'] = converted_col.dt.year
                    df[f'{col_name}_month'] = converted_col.dt.month
                    df[f'{col_name}_day'] = converted_col.dt.day
                    df.drop(columns=[col_name], inplace=True)
                    cleaning_log.append(f"SUCCESS: Split '{col_name}' into date parts and dropped original.")
                else:
                    df[col_name] = df_original[col_name]
        except Exception:
            pass # Not a datetime column

        # Alphanumeric Handling
        if df[col_name].dtype == 'object':
            col_as_str = df[col_name].astype(str)
            contains_letters = col_as_str.str.contains(r'[A-Za-z]', na=False).any()
            contains_numbers = col_as_str.str.contains(r'[0-9]', na=False).any()
            if contains_letters and contains_numbers:
                handle_alphanumeric(df, col_name, cleaning_log)

    # --- Phase 2: Reactive Cleaning ---
    profile = ProfileReport(df, minimal=True, title="Cleaned Data Profile")
    description = profile.get_description()
    alerts_list = description.alerts
    
    if alerts_list:
        for alert in alerts_list:
            if alert.alert_type == AlertType.MISSING:
                handle_missing(df, alert, description, cleaning_log)
            elif alert.alert_type == AlertType.ZEROS:
                handle_zeros(df, alert, cleaning_log)
            elif alert.alert_type == AlertType.DUPLICATES:
                handle_duplicates(df, alert, cleaning_log)

    # --- Phase 3: Outlier Detection ---
    numeric_df = df.select_dtypes(include=np.number)
    for col_name in numeric_df.columns:
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = ((df[col_name] < lower_bound) | (df[col_name] > upper_bound)).sum()
        if num_outliers > 0:
            cleaning_log.append(f"INFO: Column '{col_name}' has {num_outliers} potential outliers. No automatic action taken.")
    
    return df, cleaning_log

# ===================================================================
#  Step 3: Build the Streamlit User Interface
# ===================================================================

st.set_page_config(layout="wide")
st.title("🤖 IntelliClean: Automated Data Cleaning Tool")
st.write("Upload a CSV file and this tool will automatically clean it, handle missing values, perform feature engineering, and detect outliers.")

uploaded_file = st.file_uploader("Choose a CSV file to clean", type="csv")

if uploaded_file is not None:
    if st.button("✨ Clean My Data!"):
        with st.spinner('Running analysis and cleaning... Please wait.'):
            try:
                # Load and process the data
                df = pd.read_csv(uploaded_file)
                cleaned_df, cleaning_log = clean_and_analyze_data(df)
                
                st.success("Data cleaning complete!")

                # --- Display Results ---
                st.subheader("Cleaning Report")
                report_text = "\n".join(cleaning_log)
                st.text_area("Actions Performed:", report_text, height=300)

                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_df.head(10))

                # --- Download Button ---
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned CSV",
                    data=csv,
                    file_name=f"{uploaded_file.name.split('.')[0]}_cleaned.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please check if your CSV file is formatted correctly.")