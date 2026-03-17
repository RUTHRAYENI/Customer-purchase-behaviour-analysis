# 📊 Customer Purchase Behavior Analysis

This project is a Streamlit-based data analysis application that helps analyze customer purchasing patterns using statistical techniques, visualizations, and outlier detection methods.  

It enables users to upload their dataset and gain meaningful insights into customer behavior, spending trends, and anomalies.

---

## 🚀 Features

- Upload custom CSV datasets  
- View dataset overview and structure  
- Identify missing values and duplicates  
- Generate descriptive statistics  
- Analyze gender-based purchase distribution  
- Visualize data using:
  - Histograms  
  - Scatter plots  
  - Box plots  
- Correlation analysis with heatmap  
- Detect outliers using:
  - IQR Method  
  - Z-Score Method  
  - Isolation Forest (Machine Learning)  
- Download cleaned dataset  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Seaborn  
- Matplotlib  
- Scipy  
- Scikit-learn  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/customer-purchase-analysis.git

cd customer-purchase-analysis


### 2️⃣ Install Dependencies

pip install -r requirements.txt


### 3️⃣ Run the App

streamlit run app.py


---

## 📊 Dataset Requirements

Ensure your dataset contains the following columns:

- Age  
- Gender  
- Purchase Amount (USD)  
- Previous Purchases  
- Review Rating  

---

## 📌 Key Analysis Performed

### 🔹 Data Cleaning
- Removes missing values  
- Removes duplicate records  

### 🔹 Statistical Analysis
- Mean, Median, Mode  
- Distribution analysis  

### 🔹 Visualization
- Purchase distribution histogram  
- Age vs Purchase scatter plot  
- Gender-wise box plot  

### 🔹 Correlation
- Heatmap of numerical features  

### 🔹 Outlier Detection
- IQR Method for statistical outliers  
- Z-Score Method for standard deviation-based detection  
- Isolation Forest for machine learning-based anomaly detection  

---

## 📥 Output

- Cleaned dataset is saved as:

customer_purchases_cleaned.csv


---

## 🤝 Contributor

- Ruthrayeni A
