import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Customer Purchase Analysis", layout="wide")
st.title("Customer Purchase Behavior & Outlier Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Descriptive Statistics")
    st.write(df.describe(include='all'))
    st.write("Gender Distribution")
    st.write(df['Gender'].value_counts())

    num_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']
    st.write("Correlation Matrix")
    st.write(df[num_cols].corr())

    st.subheader("Visualizations")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Purchase Amount (USD)'], bins=20, kde=True, color='purple', ax=ax1)
    ax1.set_title("Distribution of Purchase Amounts")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Age', y='Purchase Amount (USD)', data=df, hue='Gender', s=70, ax=ax2)
    ax2.set_title("Age vs Purchase Amount by Gender")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Gender', y='Purchase Amount (USD)', data=df, ax=ax3)
    ax3.set_title("Gender-wise Purchase Distribution")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)

    col = 'Purchase Amount (USD)'
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
    sns.scatterplot(x='Age', y=col, data=df, color='blue', label='Normal', ax=ax5)
    sns.scatterplot(x='Age', y=col, data=outliers_iqr, color='red', label='Outliers', ax=ax5)
    ax5.set_title("Age vs Purchase Amount (Outliers Highlighted)")
    ax5.legend()
    st.pyplot(fig5)

    z_scores = np.abs(stats.zscore(df[col]))
    outliers_z = df[z_scores > 3]
    st.subheader("Outlier Detection (Z-Score Method)")
    st.write(f"Number of outliers detected: {len(outliers_z)}")
    st.dataframe(outliers_z)

    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Outlier_IF'] = iso.fit_predict(df[['Age', col]])
    df['Outlier_IF'] = df['Outlier_IF'].map({1:'Normal', -1:'Outlier'})
    st.subheader("Isolation Forest Outlier Detection")
    st.write(df['Outlier_IF'].value_counts())

    fig6, ax6 = plt.subplots()
    sns.scatterplot(x='Age', y=col, hue='Outlier_IF', data=df, palette={'Normal':'blue','Outlier':'red'}, s=70, ax=ax6)
    ax6.set_title("Outlier Detection using Isolation Forest")
    st.pyplot(fig6)

    st.subheader("Before vs After Removing Outliers (IQR)")
    fig7, (ax7, ax8) = plt.subplots(1,2, figsize=(12,5))
    sns.boxplot(y=df[col], color='lightcoral', ax=ax7)
    ax7.set_title("Before Removing Outliers")
    sns.boxplot(y=df_iqr_clean[col], color='lightgreen', ax=ax8)
    ax8.set_title("After Removing Outliers (IQR)")
    st.pyplot(fig7)

    mean_purchase = df[col].mean()
    median_purchase = df[col].median()
    mode_purchase = df[col].mode()[0]
    max_purchase = df[col].max()

    st.subheader("Insights Summary")
    st.write(f"Average Purchase Amount: ${mean_purchase:.2f}")
    st.write(f"Median Purchase Amount: ${median_purchase:.2f}")
    st.write(f"Most Common Purchase Amount: ${mode_purchase}")
    st.write(f"Highest Purchase Amount: ${max_purchase}")
    st.write(f"Outlier Customers (IQR): {len(outliers_iqr)} detected")

    st.subheader("Auto Insights")
    if mean_purchase < median_purchase:
        st.write("➡ Customers spend conservatively with few large purchases (left-skewed data).")
    else:
        st.write("➡ Most customers spend moderately, with few very high purchases (right-skewed data).")
    if outlier_percent > 10:
        st.write("⚠ High number of outliers detected — check for data entry errors or exceptional spenders.")
    else:
        st.write("✅ Outliers are within a reasonable range.")

    st.subheader("Download Cleaned Dataset")
    st.download_button(
        label="Download Cleaned CSV",
        data=df_iqr_clean.to_csv(index=False),
        file_name="customer_purchases_cleaned.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to start the analysis.")
