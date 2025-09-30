import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from ydata_profiling.model.alerts import AlertType
import io
import plotly.express as px
from streamlit_pandas_profiling import st_profile_report


def handling_missing(df, i, cleaned_list, alert_list):
    coln_name = i.column_name
    if coln_name not in df.columns: return
    null = i.values['p_missing']
    if null > 0.6:
        df.drop(columns=[coln_name], inplace=True)
        cleaned_list.append(f"• Dropped column '{coln_name}' due to >60% missing values.")
    else:
        if alert_list.variables[coln_name]['type'] == 'Numeric':
            median_value = df[coln_name].median()
            df[coln_name].fillna(median_value, inplace=True)
            cleaned_list.append(f"• Imputed missing values in '{coln_name}' with ({median_value}).")
        else:
            mode_val = df[coln_name].mode()[0]
            df[coln_name].fillna(mode_val, inplace=True)
            cleaned_list.append(f"• Imputed missing values in '{coln_name}' with ('{mode_val}').")

def handling_zeros(df, i, cleaned_list):
    col_name = i.column_name
    zerop = i.values['p_zeros'] * 100
    if zerop > 95:
        df[col_name] = (df[col_name] > 0).astype(int)
        cleaned_list.append(f"• Converted highly sparse column '{col_name}' to binary (0/1).")
    else:
        cleaned_list.append(f"• Column '{col_name}' has {zerop:.1f}% zeros. No automatic action taken.")

def handling_duplicates(df, i, cleaned_list):
    dup = i.values['n_duplicates']
    df.drop_duplicates(inplace=True)
    cleaned_list.append(f"• Removed {dup} duplicate rows.")
    
# Main Engine
def main_pipeline(df):
    df_copy = df.copy()
    cleaned_list = []
    correlation_report = []
    transform_candidates = []

    for i in df.columns.copy():
        if df[i].dtype == 'object':
            try:
                coln_converted = pd.to_datetime(df[i], errors='coerce')
                if coln_converted.notna().sum() / len(df) > 0.6:
                    if coln_converted.dt.month.nunique() > 1 or coln_converted.dt.day.nunique() > 1:
                        cleaned_list.append(f"• Column '{i}' was split into DateTime parts.")
                        df[f'{i}_year'] = coln_converted.dt.year; df[f'{i}_month'] = coln_converted.dt.month; df[f'{i}_day'] = coln_converted.dt.day
                        if (coln_converted.dt.normalize() != coln_converted).any(): df[f'{i}_hour'] = coln_converted.dt.hour
                        df.drop(columns=[i], inplace=True); continue
            except Exception: pass
        if i in df.columns and df[i].dtype == 'object':
            type_cabin = r'^[A-Z]\d+'; type_ticket = r'^.*?[\s]*\d+$'
            coln_sample = df[i].dropna()
            if not coln_sample.empty:
                coln_sample = coln_sample.sample(n=min(100, len(coln_sample)), random_state=1)
                if coln_sample.str.match(type_cabin).mean() > 0.6:
                    cleaned_list.append(f"• Split Column '{i}' into prefix/number.")
                    extracted = df[i].str.extract(r'^([A-Z])(\d+.*)', expand=True)
                    df[f'{i}_prefix'] = extracted[0]; df[f'{i}_number'] = pd.to_numeric(extracted[1], errors='coerce')
                    df.drop(columns=[i], inplace=True); continue
                elif coln_sample.str.match(type_ticket).mean() > 0.6:
                    cleaned_list.append(f"• Split Column '{i}' into prefix/number.")
                    col_str = df[i].astype(str)
                    df[f'{i}_prefix'] = col_str.str.replace(r'\s*\d+$', '', regex=True).str.strip().replace('', 'NONE')
                    df[f'{i}_number'] = pd.to_numeric(col_str.str.extract(r'(\d+)$')[0], errors='coerce')
                    df.drop(columns=[i], inplace=True); continue
    
    profile = ProfileReport(df, minimal=True, progress_bar=False)
    desc = profile.get_description()
    high_cooreln = {item.column_name for item in desc.alerts if item.alert_type == AlertType.HIGH_CORRELATION}
    for alert in desc.alerts:
        if alert.alert_type == AlertType.HIGH_CORRELATION:
            correlation_report.append(f"• {str(alert).replace('[', '**[').replace(']','**')}")
        elif alert.alert_type == AlertType.DUPLICATES:
            handling_duplicates(df, alert, cleaned_list)
        elif alert.alert_type == AlertType.MISSING:
            handling_missing(df, alert, cleaned_list, desc)
        elif alert.alert_type == AlertType.ZEROS:
            handling_zeros(df, alert, cleaned_list)

    num_df = df.select_dtypes(include=np.number)
    for col in num_df.columns:
        percen25 = df[col].quantile(0.25)
        percen75 = df[col].quantile(0.75)
        iqr = percen75 - percen25
        
        if iqr == 0: continue

        lower_bound, upper_bound = percen25 - (1.5 * iqr), percen75 + (1.5 * iqr)
        total_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if total_outliers > 0:
            non_na_count = df[col].notna().sum()
            if non_na_count > 0:
                outlier_percent = total_outliers / non_na_count
                if 0.02 < outlier_percent <= 0.25 and not (df[col] < 0).any():
                    transform_candidates.append(col)
    
    return df, df_copy, cleaned_list, correlation_report, transform_candidates, high_cooreln
    
# Streamlit UI 
st.set_page_config(layout="wide")
st.title("📊 IntelliClean: Automated Data Cleaning & EDA Tool")

if 'step' not in st.session_state:
    st.session_state.step = "upload"
    st.session_state.df_original = None
    st.session_state.df_cleaned = None
    st.session_state.cleaning_log = None
    st.session_state.correlation_report = None
    st.session_state.transform_candidates = None
    st.session_state.high_cooreln = None

if st.session_state.step == "upload":
    uploaded_file = st.file_uploader("Upload your CSV or JSON file", type=["csv", "json"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_json(uploaded_file)
            st.session_state.df_original = df
            if st.button("✨ Analyze & Clean Data"):
                with st.spinner("Performing initial cleaning and analysis..."):
                    cleaned, original, log, corr, candidates, high_corr_set = main_pipeline(st.session_state.df_original.copy())
                    st.session_state.df_cleaned = cleaned
                    st.session_state.cleaning_log = log
                    st.session_state.correlation_report = corr
                    st.session_state.transform_candidates = candidates
                    st.session_state.high_cooreln = high_corr_set
                    
                    if st.session_state.transform_candidates:
                        st.session_state.step = "transform_select"
                    else:
                        st.session_state.step = "view_results"
                    
                    st.rerun()
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

if st.session_state.step == "transform_select":
    st.info("Initial cleaning is complete. Please review the optional transformation below.")
    
    with st.form("transformation_form"):
        st.markdown("#### Optional: Log Transformation for Skewed Data")
        st.write("The following columns have been identified as having 2-15% outliers. **De-select any columns you wish to exclude** from the log transformation.")
        
        correlated_candidates = [col for col in st.session_state.transform_candidates if col in st.session_state.high_cooreln]
        if correlated_candidates:
            st.warning(f"**Warning:** The following candidates are also flagged as highly correlated and transforming them may not be advisable: **{', '.join(correlated_candidates)}**")

        selected_to_transform = st.multiselect(
            "Select columns to transform:",
            options=st.session_state.transform_candidates,
            default=st.session_state.transform_candidates
        )
        
        submitted = st.form_submit_button("✅ Apply Selections & View Results")
        if submitted:
            if selected_to_transform:
                for col in selected_to_transform:
                    st.session_state.df_cleaned[col] = np.log1p(st.session_state.df_cleaned[col])
                    st.session_state.cleaning_log.append(f"• Applied user-selected log transformation to '{col}'.")
            st.session_state.step = "view_results"
            st.rerun()

if st.session_state.step == "view_results":
    st.success("All cleaning and transformations are complete!")
    
    st.markdown('<h3 style="color: #0072B2;">Final Cleaning Report</h3>', unsafe_allow_html=True)
    st.text_area("Actions Performed:", "\n".join(st.session_state.cleaning_log), height=300)

    st.markdown('<h3 style="color: #0072B2;">Data Comparison</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Original Shape", f"{st.session_state.df_original.shape[0]} rows, {st.session_state.df_original.shape[1]} cols")
    col2.metric("Cleaned Shape", f"{st.session_state.df_cleaned.shape[0]} rows, {st.session_state.df_cleaned.shape[1]} cols")
    
    tab1, tab2 = st.tabs(["Cleaned Data Preview", "Original Data"])
    with tab1:
        st.dataframe(st.session_state.df_cleaned)
    with tab2:
        st.dataframe(st.session_state.df_original)

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
                selected_num_col = st.selectbox("Select a NUMERIC column for Histograms:", options=[""] + common_numeric_cols, key="num_select")
            with v_col2:
                selected_cat_col = st.selectbox("Select a CATEGORICAL column for Bar Charts:", options=[""] + common_categorical_cols, key="cat_select")

            if selected_num_col:
                #  Added a spinner 
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
    
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv = st.session_state.df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")
    with dl_col2:
        json_data = st.session_state.df_cleaned.to_json(orient='records', indent=4).encode('utf-8')
        st.download_button(label="Download as JSON", data=json_data, file_name="cleaned_data.json", mime="application/json")
    
    if st.button("↩️ Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
