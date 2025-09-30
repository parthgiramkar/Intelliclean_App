import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from ydata_profiling.model.alerts import AlertType
import io

# ===================================================================
#  Step 1: Define All Your Cleaning and Handling Functions
# ===================================================================

def handle_missing(df, alert, description, log):
    col_name = alert.column_name
    if col_name not in df.columns:
        return
    p_missing = alert.values['p_missing']

    if p_missing > 0.6:
        df.drop(columns=[col_name], inplace=True)
        log.append(f"• Dropped column '{col_name}' due to >60% missing values.")
    else:
        if description.variables[col_name]['type'] == 'Numeric':
            median_val = df[col_name].median()
            df[col_name].fillna(median_val, inplace=True)
            log.append(f"• Imputed missing values in '{col_name}' with ({median_val}).")
        else: 
            mode_val = df[col_name].mode()[0]
            df[col_name].fillna(mode_val, inplace=True)
            log.append(f"• Imputed missing values in '{col_name}' with ('{mode_val}').")

def handle_zeros(alert, log):
    col_name = alert.column_name
    p_zeros = alert.values['p_zeros']
    log.append(f"• Column '{col_name}' has {p_zeros:.1%} zeros.")

def handle_duplicates(df, alert, log):
    num_duplicates = alert.values['n_duplicates']
    df.drop_duplicates(inplace=True)
    log.append(f"• Removed {num_duplicates} duplicate rows.")

# ===================================================================
#  Step 2: A main "engine" function that runs the whole pipeline
# ===================================================================

def run_cleaning_pipeline(df):
    cleaning_log = []
    original_df = df.copy()

    # --- Phase 1: Proactive Transformations ---
    for col_name in df.columns.copy():
      if df[col_name].dtype == 'object':
        # Datetime Handling
        try:
            converted_col = pd.to_datetime(df[col_name], errors='coerce')
            if converted_col.notna().sum() / len(df) > 0.6:
                if converted_col.dt.month.nunique() > 1 or converted_col.dt.day.nunique() > 1:
                    cleaning_log.append(f"• Column '{col_name}' was split into DateTime parts.")
                    df[f'{col_name}_year'] = converted_col.dt.year
                    df[f'{col_name}_month'] = converted_col.dt.month
                    df[f'{col_name}_day'] = converted_col.dt.day
                    if (converted_col.dt.normalize() != converted_col).any():
                        df[f'{col_name}_hour'] = converted_col.dt.hour
                    df.drop(columns=[col_name], inplace=True)
                    continue
        except Exception:
            pass 

        # Alphanumeric Handling
        if df[col_name].dtype == 'object':
            cabin_pattern = r'^[A-Z]\d+'
            ticket_pattern = r'^.*?[\s]*\d+$'
            col_sample = df[col_name].dropna()
            if not col_sample.empty:
                col_sample = col_sample.sample(n=min(100, len(col_sample)), random_state=1)
                cabin_match_pct = col_sample.str.match(cabin_pattern).mean()
                ticket_match_pct = col_sample.str.match(ticket_pattern).mean()

                if cabin_match_pct > 0.6:
                    extracted = df[col_name].str.extract(r'^([A-Z])(\d+.*)', expand=True)
                    df[f'{col_name}_prefix'] = extracted[0]
                    df[f'{col_name}_number'] = pd.to_numeric(extracted[1], errors='coerce')
                    df.drop(columns=[col_name], inplace=True)
                    cleaning_log.append(f"• Split Column '{col_name}' into prefix/number parts.")
                    continue
                elif ticket_match_pct > 0.6:
                    col_as_str = df[col_name].astype(str)
                    df[f'{col_name}_prefix'] = col_as_str.str.replace(r'\s*\d+$', '', regex=True).str.strip()
                    df[f'{col_name}_prefix'].replace('', 'NONE', inplace=True)
                    df[f'{col_name}_number'] = pd.to_numeric(col_as_str.str.extract(r'(\d+)$')[0], errors='coerce')
                    df.drop(columns=[col_name], inplace=True)
                    cleaning_log.append(f"• Split Column '{col_name}' into prefix/number parts.")
                    continue

    # --- Phase 2: Reactive Cleaning ---
    profile = ProfileReport(df, minimal=True)
    description = profile.get_description()
    alerts_list = description.alerts
    
    correlated_cols = {alert.column_name for alert in alerts_list if alert.alert_type == AlertType.HIGH_CORRELATION}
    
    if alerts_list:
        for alert in alerts_list:
            if alert.alert_type == AlertType.DUPLICATES:
                handle_duplicates(df, alert, cleaning_log)
            elif alert.alert_type == AlertType.MISSING:
                if alert.column_name in correlated_cols:
                    cleaning_log.append(f"• Missing values in '{alert.column_name}' were NOT imputed due to high correlation.")
                    continue
                handle_missing(df, alert, description, cleaning_log)
            elif alert.alert_type == AlertType.ZEROS:
                handle_zeros(alert, cleaning_log)

    # --- Phase 3: Outlier Detection ---
    numeric_df = df.select_dtypes(include=np.number)
    for col_name in numeric_df.columns:
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_b, upper_b = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        num_outliers = ((df[col_name] < lower_b) | (df[col_name] > upper_b)).sum()

        percen = num_outliers/len(df.dropna(subset=[col_name]))

        if num_outliers > 0 :

            if percen >= 0.2 and percen <= 0.15 :
                 
                if (df[col_name] <= 0).any() :

                    cleaning_log.append(f"• {col_name} is highly skewed but was not log-transformed as it contains non-positive values.")

                else :
                    df[col_name] = np.log1p(df[col_name] ) 

                    cleaning_log.append(f"• Applied Log tranformation to column {col_name} to handle high skewness.")

            else :

                cleaning_log.append(f"• {col_name} has '{num_outliers}' potential  outliers.")

        
    return df, original_df, cleaning_log

# ===================================================================
#  Step 3: The Streamlit User Interface
# ===================================================================
# ===================================================================
#  The Streamlit User Interface (with color changes)
# ===================================================================

st.set_page_config(layout="wide")
st.title("📊 IntelliClean: Automated Data Cleaning & EDA Tool")


uploaded_file = st.file_uploader("Upload your CSV file to begin", type="csv")

if uploaded_file is not None:
    if st.button("✨ Clean My Data!"):
        with st.spinner('Running analysis and cleaning... Please wait.'):
            try:
                input_df = pd.read_csv(uploaded_file)
                
                cleaned_df, original_df, cleaning_log = run_cleaning_pipeline(input_df)
                
                st.success("Cleaning complete!")

                # --- Use markdown for a colored subheader ---
                st.markdown('<h3 style="color: #0072B2;">Cleaning Report</h3>', unsafe_allow_html=True)
                report_text = "\n".join(cleaning_log)
                st.text_area("Actions Performed - ", report_text, height=300)

                # --- Use markdown for a colored subheader ---
                st.markdown('<h3 style="color: #0072B2;">Data Comparison</h3>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Original Shape", f"{original_df.shape[0]} rows, {original_df.shape[1]} cols")
                col2.metric("Cleaned Shape", f"{cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} cols")

                # The tabs will now be colored by the CSS block above
                tab1, tab2 = st.tabs(["Cleaned Data Preview", "Original Data"])
                with tab1:
                    st.dataframe(cleaned_df)
                with tab2:
                    st.dataframe(original_df)
                
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned CSV",
                    data=csv,
                    file_name=f"cleaned_{uploaded_file.name}",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred during the cleaning process: {e}")