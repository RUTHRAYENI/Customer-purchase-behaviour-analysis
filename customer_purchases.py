# ==============================================
# Streamlit App: Customer Purchase Behavior Analysis
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

# Streamlit page config
st.set_page_config(page_title="Customer Purchase Analysis", layout="wide")

# -----------------------------
# 1️⃣ Title & Upload Dataset
# -----------------------------
st.title("📊 Customer Purchase Behavior Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("✅ First 5 Records")
    st.dataframe(df.head())

    # -----------------------------
    # 2️⃣ Data Overview
    # -----------------------------
    st.subheader("📋 Dataset Information")
    buffer = df.info(buf=None)
    st.text(df.info())

    st.subheader("🔍 Missing Values")
    st.dataframe(df.isnull().sum())

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # -----------------------------
    # 3️⃣ Descriptive Statistics
    # -----------------------------
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe(include='all'))

    st.subheader("⚖ Gender Distribution")
    st.bar_chart(df['Gender'].value_counts())

    # -----------------------------
    # 4️⃣ Correlation
    # -----------------------------
    numeric_cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']
    st.subheader("📌 Correlation Matrix")
    corr = df[numeric_cols].corr()
    st.dataframe(corr)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # 5️⃣ Visualizations
    # -----------------------------
    st.subheader("📊 Purchase Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Purchase Amount (USD)'], bins=20, kde=True, color='purple', ax=ax)
    ax.set_xlabel("Purchase Amount (USD)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("👥 Age vs Purchase Amount by Gender")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='Purchase Amount (USD)', hue='Gender', data=df, s=70, ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Gender-wise Purchase Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Purchase Amount (USD)', data=df, ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # 6️⃣ Outlier Detection - IQR Method
    # -----------------------------
    col = 'Purchase Amount (USD)'
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_percent = (len(outliers_iqr)/len(df)) * 100
    df_iqr_clean = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    st.subheader("⚠ Outlier Detection (IQR Method)")
    st.write(f"Number of outliers: {len(outliers_iqr)} ({outlier_percent:.2f}%)")

    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y=col, data=df, color='blue', label='Normal', ax=ax)
    sns.scatterplot(x='Age', y=col, data=outliers_iqr, color='red', label='Outlier', ax=ax)
    ax.set_title("Age vs Purchase Amount (Outliers Highlighted)")
    st.pyplot(fig)

    # -----------------------------
    # 7️⃣ Outlier Detection - Z-Score Method
    # -----------------------------
    z_scores = np.abs(stats.zscore(df[col]))
    threshold = 3
    outliers_z = df[z_scores > threshold]
    st.write(f"Number of outliers detected (Z-Score > {threshold}): {len(outliers_z)}")

    # -----------------------------
    # 8️⃣ Isolation Forest Outlier Detection
    # -----------------------------
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['Outlier_IF'] = iso.fit_predict(df[['Age', col]])
    df['Outlier_IF'] = df['Outlier_IF'].map({1: 'Normal', -1: 'Outlier'})

    st.subheader("Isolation Forest Outlier Detection")
    st.dataframe(df['Outlier_IF'].value_counts())

    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y=col, hue='Outlier_IF', data=df,
                    palette={'Normal':'blue', 'Outlier':'red'}, s=70, ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # 9️⃣ Summary Insights
    # -----------------------------
    mean_purchase = df[col].mean()
    median_purchase = df[col].median()
    mode_purchase = df[col].mode()[0]
    max_purchase = df[col].max()

    st.subheader("📌 Insights Summary")
    st.write(f"Average Purchase Amount: ${mean_purchase:.2f}")
    st.write(f"Median Purchase Amount: ${median_purchase:.2f}")
    st.write(f"Most Common Purchase Amount: ${mode_purchase}")
    st.write(f"Highest Purchase Amount: ${max_purchase}")
    st.write(f"Outlier Customers (IQR): {len(outliers_iqr)} detected")

    # -----------------------------
    # 10️⃣ Save Cleaned Dataset
    # -----------------------------
    df_iqr_clean.to_csv("customer_purchases_cleaned.csv", index=False)
