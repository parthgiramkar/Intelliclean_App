import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from ydata_profiling.model.alerts import AlertType
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px

# ===================================================================
# Function Definitions
# ===================================================================
def handling_missing(df, alert, cleaned_list, desc):
    col_name = alert.column_name
    if col_name not in df.columns: return
    missing_pct = alert.values.get('p_missing', 0)  # Safer access with .get()
    if missing_pct > 0.6:
        df.drop(columns=[col_name], inplace=True)
        cleaned_list.append(f"• Dropped column '{col_name}' due to >60% missing values ({missing_pct:.1%}).")
    elif missing_pct > 0:
        col_type = desc.variables.get(col_name, {}).get('type', None)
        if col_type == 'Numeric':
            median_value = df[col_name].median()
            df[col_name].fillna(median_value, inplace=True)
            cleaned_list.append(f"• Imputed missing values in '{col_name}' with median ({median_value}).")
        else:
            mode_val = df[col_name].mode()[0] if not df[col_name].mode().empty else np.nan
            df[col_name].fillna(mode_val, inplace=True)
            cleaned_list.append(f"• Imputed missing values in '{col_name}' with mode ('{mode_val}').")

def handling_zeros(df, alert, cleaned_list):
    col_name = alert.column_name
    zeros_pct = alert.values.get('p_zeros', 0) * 100  # Safer access
    if zeros_pct > 95:
        df[col_name] = (df[col_name] > 0).astype(int)
        cleaned_list.append(f"• Converted highly sparse column '{col_name}' to binary (0/1) due to {zeros_pct:.1f}% zeros.")
    else:
        cleaned_list.append(f"• Column '{col_name}' has {zeros_pct:.1f}% zeros. No automatic action taken.")

# ===================================================================
# Main Cleaning Pipeline Function (Improved)
# ===================================================================
def main_pipeline(df, outlier_method="None", excluded_cols=None):
    if excluded_cols is None: excluded_cols = []
    df_copy = df.copy()
    cleaned_list = []
    correlation_report = []
    
    # STEP 1: Run analysis on original data with explorative mode for full alerts
    profile = ProfileReport(df, explorative=True, progress_bar=False)  # Use explorative for detailed alerts
    desc = profile.get_description()
    
    # STEP 2: Handle Duplicates
    dup_count = desc.table.get('n_duplicates', 0)
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        cleaned_list.append(f"• Removed {dup_count} duplicate rows.")
    
    # STEP 3: Proactive Transformations (unchanged, but moved before alerts to avoid interference)
    for col in df.columns.copy():
        if df[col].dtype == 'object':
            # Attempt datetime conversion
            col_converted = pd.to_datetime(df[col], errors='coerce')
            if col_converted.notna().sum() / len(df) > 0.6 and col_converted.dt.normalize().nunique() > 1:
                cleaned_list.append(f"• Column '{col}' was split into DateTime parts.")
                df[f'{col}_year'] = col_converted.dt.year
                df[f'{col}_month'] = col_converted.dt.month
                df[f'{col}_day'] = col_converted.dt.day
                if (col_converted.dt.normalize() != col_converted).any():
                    df[f'{col}_hour'] = col_converted.dt.hour
                df.drop(columns=[col], inplace=True)
                continue
            
            # Cabin/Ticket splitting (refined regex)
            if col in df.columns and df[col].dtype == 'object':
                sample = df[col].dropna().sample(n=min(100, len(df[col].dropna())), random_state=1)
                if sample.str.match(r'^[A-Z]\d+').mean() > 0.6:  # Cabin-like
                    cleaned_list.append(f"• Split Cabin-like column '{col}' into prefix/number.")
                    extracted = df[col].str.extract(r'^([A-Z])(\d+.*)', expand=True)
                    df[f'{col}_prefix'] = extracted[0]
                    df[f'{col}_number'] = pd.to_numeric(extracted[1], errors='coerce')
                    df.drop(columns=[col], inplace=True)
                    continue
                elif sample.str.match(r'^(\S+\s)?\S*\d+$').mean() > 0.6:  # Ticket-like
                    cleaned_list.append(f"• Split Ticket-like column '{col}' into prefix/number.")
                    col_str = df[col].astype(str)
                    df[f'{col}_prefix'] = col_str.str.replace(r'\s*\d+$', '', regex=True).str.strip().replace('', 'NONE')
                    df[f'{col}_number'] = pd.to_numeric(col_str.str.extract(r'(\d+)$')[0], errors='coerce')
                    df.drop(columns=[col], inplace=True)
                    continue
    
    # STEP 4: Reactive Cleaning based on Alerts (improved handling)
    if not desc.alerts:
        cleaned_list.append("• No alerts generated by ydata-profiling. Ensure explorative mode is enabled.")
    for alert in desc.alerts:
        if alert.alert_type == AlertType.HIGH_CORRELATION:
            correlation_report.append(f"• {str(alert).replace('[', '**[').replace(']','**')}")
        elif alert.alert_type == AlertType.MISSING:
            handling_missing(df, alert, cleaned_list, desc)
        elif alert.alert_type == AlertType.ZEROS:
            handling_zeros(df, alert, cleaned_list)
        elif alert.alert_type == AlertType.SKEWNESS and outlier_method == "Log Transform":
            # Optional: Use skewness alert to trigger log transform
            col_name = alert.column_name
            if col_name not in excluded_cols and df[col_name].min() >= 0:
                df[col_name] = np.log1p(df[col_name])
                cleaned_list.append(f"• Applied log transformation to '{col_name}' due to skewness alert.")
        else:
            cleaned_list.append(f"• Unhandled alert: {alert.alert_type} for '{alert.column_name}'.")
    
    # STEP 5: Advanced Outlier Transformation (as fallback if no skewness alert)
    if outlier_method == "Log Transform":
        cleaned_list.append("--- Applying Log Transformation for Skewness (Fallback) ---")
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col in excluded_cols: 
                cleaned_list.append(f"• Skipped log transformation for '{col}' as per user exclusion.")
                continue
            q25, q75 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q75 - q25
            if iqr == 0: continue
            outliers = ((df[col] < q25 - 1.5 * iqr) | (df[col] > q75 + 1.5 * iqr)).sum()
            if outliers > 0 and (outliers / df[col].notna().sum()) > 0.02 and df[col].min() >= 0:
                df[col] = np.log1p(df[col])
                cleaned_list.append(f"• Applied log transformation to '{col}' to handle outliers.")
    
    return df, df_copy, cleaned_list, correlation_report

# ===================================================================
# Streamlit User Interface
# ===================================================================
st.set_page_config(layout="wide")
st.title("📊 IntelliClean: Automated Data Cleaning & EDA Tool")
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
    st.session_state.df_original = None
    st.session_state.cleaning_log = None
    st.session_state.correlation_report = None
    st.session_state.numeric_columns = []
uploaded_file = st.file_uploader("Upload your CSV or JSON file", type=["csv", "json"])
if uploaded_file is None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
if uploaded_file:
    st.sidebar.header("⚙️ Cleaning Configuration")
    outlier_method = st.sidebar.selectbox(
        "Handle Skewed Numeric Data:",
        ("None", "Log Transform"),
        help="Choose 'Log Transform' to fix skewed data."
    )
    df_original = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
    numeric_columns = df_original.select_dtypes(include=np.number).columns.tolist()
    excluded_cols = st.sidebar.multiselect(
        "Columns to Exclude from Transformation:",
        options=numeric_columns,
        help="Select columns to protect from log transformation."
    )
    if st.button("✨ Analyze & Clean Data"):
        with st.spinner("Performing cleaning and analysis..."):
            st.session_state.df_original = df_original
            cleaned, original, log, corr = main_pipeline(
                st.session_state.df_original.copy(), 
                outlier_method=outlier_method,
                excluded_cols=excluded_cols
            )
            st.session_state.df_cleaned = cleaned
            st.session_state.cleaning_log = log
            st.session_state.correlation_report = corr
            st.success("Cleaning complete!")
if st.session_state.get('df_cleaned') is not None:
    st.markdown('<h3 style="color: #0072B2;">Final Cleaning Report</h3>', unsafe_allow_html=True)
    st.text_area("Actions Performed:", "\n".join(st.session_state.cleaning_log), height=250)
    st.markdown('<h3 style="color: #0072B2;">Data Comparison</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Original Shape", f"{st.session_state.df_original.shape[0]} rows, {st.session_state.df_original.shape[1]} cols")
    col2.metric("Cleaned Shape", f"{st.session_state.df_cleaned.shape[0]} rows, {st.session_state.df_cleaned.shape[1]} cols")
    
    tab1, tab2 = st.tabs(["Cleaned Data Preview", "Original Data"])
    with tab1:
        st.dataframe(st.session_state.df_cleaned)
    with tab2:
        st.dataframe(st.session_state.df_original)
    
    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV", 
            data=csv, 
            file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.csv", 
            mime="text/csv"
        )
    with dl_col2:
        json_data = st.session_state.df_cleaned.to_json(orient='records', indent=4).encode('utf-8')
        st.download_button(
            label="Download as JSON", 
            data=json_data, 
            file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.json", 
            mime="application/json"
        )
    st.markdown("---")
    st.markdown('<h3 style="color: #0072B2;">Before & After Visualizations</h3>', unsafe_allow_html=True)
    with st.expander("📊 Click to see column comparisons"):
        common_cols = list(set(st.session_state.df_original.columns) & set(st.session_state.df_cleaned.columns))
        common_numeric_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(st.session_state.df_original[col])]
        common_categorical_cols = [col for col in common_cols if pd.api.types.is_object_dtype(st.session_state.df_original[col])]
        if not common_numeric_cols and not common_categorical_cols:
            st.warning("No common columns available for visualization.")
        else:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                selected_num_col = st.selectbox("Select a NUMERIC column for Histograms:", options=[""] + common_numeric_cols)
            with v_col2:
                selected_cat_col = st.selectbox("Select a CATEGORICAL column for Bar Charts:", options=[""] + common_categorical_cols)
            
            if selected_num_col:
                with st.spinner(f"Generating charts for '{selected_num_col}'..."):
                    st.subheader(f"Distribution of '{selected_num_col}'")
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_before = px.histogram(st.session_state.df_original, x=selected_num_col, title="Before Cleaning")
                        st.plotly_chart(fig_before, use_container_width=True)
                    with c2:
                        fig_after = px.histogram(st.session_state.df_cleaned, x=selected_num_col, title="After Cleaning")
                        st.plotly_chart(fig_after, use_container_width=True)
            
            if selected_cat_col:
                with st.spinner(f"Generating charts for '{selected_cat_col}'..."):
                    st.subheader(f"Value Counts of '{selected_cat_col}'")
                    unique_count = st.session_state.df_original[selected_cat_col].nunique()
                    if unique_count > 30:
                        st.info(f"Showing the Top 20 most frequent values for '{selected_cat_col}' (Total Unique Values: {unique_count}).")
                        c1, c2 = st.columns(2)
                        with c1:
                            before_counts = st.session_state.df_original[selected_cat_col].value_counts().head(20).reset_index()
                            before_counts.columns = [selected_cat_col, 'count']
                            fig_before = px.bar(before_counts, x=selected_cat_col, y='count', title="Before Cleaning (Top 20)")
                            st.plotly_chart(fig_before, use_container_width=True)
                        with c2:
                            after_counts = st.session_state.df_cleaned[selected_cat_col].value_counts().head(20).reset_index()
                            after_counts.columns = [selected_cat_col, 'count']
                            fig_after = px.bar(after_counts, x=selected_cat_col, y='count', title="After Cleaning (Top 20)")
                            st.plotly_chart(fig_after, use_container_width=True)
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            before_counts = st.session_state.df_original[selected_cat_col].value_counts().reset_index()
                            before_counts.columns = [selected_cat_col, 'count']
                            fig_before = px.bar(before_counts, x=selected_cat_col, y='count', title="Before Cleaning")
                            st.plotly_chart(fig_before, use_container_width=True)
                        with c2:
                            after_counts = st.session_state.df_cleaned[selected_cat_col].value_counts().reset_index()
                            after_counts.columns = [selected_cat_col, 'count']
                            fig_after = px.bar(after_counts, x=selected_cat_col, y='count', title="After Cleaning")
                            st.plotly_chart(fig_after, use_container_width=True)
    
    st.markdown('<h3 style="color: #0072B2;">Exploratory Data Analysis</h3>', unsafe_allow_html=True)
    with st.expander("🔍 Click to view the detailed Interactive EDA Report"):
        eda_tab1, eda_tab2 = st.tabs(["📊 After Cleaning", "📄 Before Cleaning"])
        with eda_tab1:
            with st.spinner("Generating 'After Cleaning' report... Please wait."):
                report_config = {"title": "Cleaned Data Profile", "html": {"style": {"theme": "flatly", "show_powered_by": False}}}
                cleaned_profile = ProfileReport(st.session_state.df_cleaned, **report_config)
                st_profile_report(cleaned_profile)
        with eda_tab2:
            with st.spinner("Generating 'Before Cleaning' report... Please wait."):
                report_config = {"title": "Original Data Profile", "html": {"style": {"theme": "flatly", "show_powered_by": False}}}
                original_profile = ProfileReport(st.session_state.df_original, **report_config)
                st_profile_report(original_profile)
    
    if st.session_state.correlation_report:
        st.subheader("Correlation Analysis")
        with st.expander("🔍 Click to see highly correlated columns"):
            st.markdown("\n".join(st.session_state.correlation_report))
