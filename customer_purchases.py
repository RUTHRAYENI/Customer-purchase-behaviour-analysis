import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Customer Purchase Analysis", layout="wide")
st.title("Customer Purchase Behavior & Outlier Analysis")

st.write("Upload a CSV file containing customer purchase data with at least 'Age' and 'Purchase Amount' columns.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Columns in Dataset")
    st.write(df.columns.tolist())

    # Automatically detect numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Numeric Columns Used for Analysis")
    st.write(num_cols)

    st.subheader("Descriptive Statistics")
    st.write(df.describe(include='all'))

    if 'Gender' in df.columns:
        st.write("Gender Distribution")
        st.write(df['Gender'].value_counts())

    if len(num_cols) >= 2:
        st.subheader("Correlation Matrix")
        st.write(df[num_cols].corr())

        # Visualizations
        st.subheader("Visualizations")

        # Histogram of first numeric column
        col = num_cols[0]
        fig1, ax1 = plt.subplots()
        sns.histplot(df[col], bins=20, kde=True, color='purple', ax=ax1)
        ax1.set_title(f"Distribution of {col}")
        st.pyplot(fig1)

        # Scatter plot: first vs second numeric column
        fig2, ax2 = plt.subplots()
        hue_col = 'Gender' if 'Gender' in df.columns else None
        sns.scatterplot(x=num_cols[0], y=num_cols[1], data=df, hue=hue_col, s=70, ax=ax2)
        ax2.set_title(f"{num_cols[0]} vs {num_cols[1]}" + (f" by {hue_col}" if hue_col else ""))
        st.pyplot(fig2)

        # Boxplot for first numeric column grouped by Gender
        if 'Gender' in df.columns:
            fig3, ax3 = plt.subplots()
            sns.boxplot(x='Gender', y=num_cols[0], data=df, ax=ax3)
            ax3.set_title(f"{num_cols[0]} by Gender")
            st.pyplot(fig3)

        # Correlation heatmap
        fig4, ax4 = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
        ax4.set_title("Correlation Heatmap")
        st.pyplot(fig4)

        # -----------------------------
        # Outlier Detection - IQR Method
        # -----------------------------
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        df_iqr_clean = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        outlier_percent = (len(outliers_iqr)/len(df))*100

        st.subheader("Outlier Detection (IQR Method)")
        st.write(f"Number of outliers detected: {len(outliers_iqr)} ({outlier_percent:.2f}%)")
        st.dataframe(outliers_iqr)

        fig5, ax5 = plt.subplots()
        sns.scatterplot(x=num_cols[0], y=col, data=df, color='blue', label='Normal', ax=ax5)
        sns.scatterplot(x=num_cols[0], y=col, data=outliers_iqr, color='red', label='Outliers', ax=ax5)
        ax5.set_title(f"{num_cols[0]} vs {col} (Outliers Highlighted)")
        ax5.legend()
        st.pyplot(fig5)

        # Z-Score method
        z_scores = np.abs(stats.zscore(df[col]))
        outliers_z = df[z_scores > 3]
        st.subheader("Outlier Detection (Z-Score Method)")
        st.write(f"Number of outliers detected: {len(outliers_z)}")
        st.dataframe(outliers_z)

        # Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['Outlier_IF'] = iso.fit_predict(df[[num_cols[0], num_cols[1]]])
        df['Outlier_IF'] = df['Outlier_IF'].map({1:'Normal', -1:'Outlier'})
        st.subheader("Isolation Forest Outlier Detection")
        st.write(df['Outlier_IF'].value_counts())

        fig6, ax6 = plt.subplots()
        sns.scatterplot(x=num_cols[0], y=col, hue='Outlier_IF', data=df, palette={'Normal':'blue','Outlier':'red'}, s=70, ax=ax6)
        ax6.set_title("Outlier Detection using Isolation Forest")
        st.pyplot(fig6)

        # Boxplots before and after IQR removal
        st.subheader("Before vs After Removing Outliers (IQR)")
        fig7, (ax7, ax8) = plt.subplots(1,2, figsize=(12,5))
        sns.boxplot(y=df[col], color='lightcoral', ax=ax7)
        ax7.set_title("Before Removing Outliers")
        sns.boxplot(y=df_iqr_clean[col], color='lightgreen', ax=ax8)
        ax8.set_title("After Removing Outliers (IQR)")
        st.pyplot(fig7)

        # Summary
        mean_purchase = df[col].mean()
        median_purchase = df[col].median()
        mode_purchase = df[col].mode()[0]
        max_purchase = df[col].max()

        st.subheader("Insights Summary")
        st.write(f"Average {col}: {mean_purchase:.2f}")
        st.write(f"Median {col}: {median_purchase:.2f}")
        st.write(f"Most Common {col}: {mode_purchase}")
        st.write(f"Highest {col}: {max_purchase}")
        st.write(f"Outlier Customers (IQR): {len(outliers_iqr)} detected")

        st.subheader("Auto Insights")
        if mean_purchase < median_purchase:
            st.write(f"➡ Customers spend conservatively with few large purchases (left-skewed {col}).")
        else:
            st.write(f"➡ Most customers spend moderately, with few very high purchases (right-skewed {col}).")
        if outlier_percent > 10:
            st.write("⚠ High number of outliers detected — check for data entry errors or exceptional spenders.")
        else:
            st.write("✅ Outliers are within a reasonable range.")

        # Download cleaned dataset
        st.subheader("Download Cleaned Dataset")
        st.download_button(
            label="Download Cleaned CSV",
            data=df_iqr_clean.to_csv(index=False),
            file_name="customer_purchases_cleaned.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to start the analysis.")

