# Intelliclean: Automated Data Cleaning & EDA Tool 📊✨

Intelliclean is an interactive web application built with **Streamlit** that automates the initial, often tedious, steps of data cleaning and exploratory data analysis. Upload your dataset, configure the cleaning options, and get a clean dataset and a comprehensive EDA report in minutes.



---

## ## 🚀 Features

Intelliclean provides a robust pipeline for cleaning and understanding your data, combining automated actions with essential user controls.

#### ### Automated Cleaning Pipeline
* **Duplicate Removal:** Automatically detects and removes duplicate rows.
* **Missing Value Imputation:**
    * Fills missing numeric data with the **median**.
    * Fills missing categorical data with the **mode**.
    * Drops columns that exceed a high threshold of missing values (e.g., >60%).
* **Zero-Value Handling:** Converts numeric columns with a very high percentage of zeros (e.g., >95%) into a binary (0/1) format.

#### ### Proactive Transformations
* **Smart DateTime Splitting:** Automatically detects columns that are dates (not just times) and splits them into `year`, `month`, `day`, and `hour` components.
* **Pattern-Based Splitting:** Identifies and splits complex string columns that match common patterns, such as ticket or cabin numbers (e.g., 'C85' -> prefix 'C', number '85').

#### ### User-Controlled Outlier Handling
* **Optional Log Transformation:** Users can choose to apply a log transformation (`log1p`) to fix numeric columns that are identified as having high skewness.
* **Manual Column Exclusion:** Provides a multi-select dropdown in the sidebar for the user to **protect** specific columns (like `Age`, `ID`, or `longitude`) from being automatically transformed.

#### ### Interactive Visualizations & Reports
* **"Before & After" Plots:** Instantly visualize the impact of cleaning operations with side-by-side distribution plots for any numeric or categorical column.
* **Full EDA Report:** Generates a comprehensive and interactive EDA report for both the original and the cleaned dataframes, powered by `ydata-profiling`.
* **Correlation Summary:** Displays a list of highly correlated columns identified during the analysis.

#### ### Data Export
* **Download Cleaned Data:** Easily download the final, cleaned dataset as a **CSV** or **JSON** file.

---

## ## 🛠️ Tech Stack

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application UI.
* **Pandas:** For all data manipulation and cleaning operations.
* **NumPy:** For numerical operations, including the log transformation.
* **Plotly Express:** For generating the "Before & After" visualization charts.
* **ydata-profiling:** For generating the comprehensive EDA reports.

---

## ## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-link>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On MacOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Make sure your `requirements.txt` file includes `streamlit`, `pandas`, `numpy`, `ydata-profiling`, `streamlit-pandas-profiling`, and `plotly`)*

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    *(Assuming your main Python script is named `app.py`)*

---

## ## 📖 How to Use

1.  Launch the application.
2.  Upload your CSV or JSON file using the file uploader.
3.  Use the sidebar to configure the cleaning options (e.g., enable log transform, exclude specific columns).
4.  Click the **"Analyze & Clean Data"** button.
5.  Review the cleaning log, data previews, and interactive visualizations.
6.  Download your cleaned dataset.

---

## ## 📄 License

This project is licensed under the MIT License.
