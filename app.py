import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Logger ----------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# ---------------- Session State ----------------
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ---------------- Folder Setup ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

log("Application started")

# ---------------- Page Config ----------------
st.set_page_config("End-to-End Naive Bayes", layout="wide")
st.title("End-to-End Naive Bayes Platform")

# ---------------- Sidebar ----------------
st.sidebar.header("Naive Bayes Settings")
nb_type = st.sidebar.selectbox(
    "Naive Bayes Variant",
    ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"]
)

# ---------------- Step 1: Data Ingestion ----------------
st.header("Step 1: Data Ingestion")
option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])

df = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        raw_path = os.path.join(RAW_DIR, "iris.csv")

        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)
        st.success("Dataset downloaded successfully")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")

# ---------------- Step 2: EDA ----------------
if df is not None:
    st.header("Step 2: EDA")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isnull().sum())

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ---------------- Step 3: Data Cleaning ----------------
if df is not None:
    st.header("Step 3: Data Cleaning")
    strategy = st.selectbox("Missing value strategy", ["Mean", "Median", "Drop Rows"])

    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.session_state.df_clean = df_clean
    st.success("Data cleaning completed")

# ---------------- Step 4: Save Cleaned Data ----------------
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("Please clean data first")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_path = os.path.join(CLEANED_DIR, f"cleaned_{timestamp}.csv")
        st.session_state.df_clean.to_csv(clean_path, index=False)
        st.success("Cleaned dataset saved")
        st.info(clean_path)
        log(f"Cleaned dataset saved at {clean_path}")

# ---------------- Step 5: Load Cleaned Dataset ----------------
st.header("Step 5: Load Cleaned Dataset")
files = os.listdir(CLEANED_DIR)

if files:
    selected = st.selectbox("Select Dataset", files)
    df_model = pd.read_csv(os.path.join(CLEANED_DIR, selected))
    st.dataframe(df_model.head())
else:
    st.warning("No cleaned datasets found")
    st.stop()

# ---------------- Step 6: Train Naive Bayes ----------------
st.header("Step 6: Train Naive Bayes")

# ✔ Restrict target to categorical columns
categorical_cols = df_model.select_dtypes(include=["object"]).columns
if len(categorical_cols) == 0:
    st.error("No categorical target column found for classification")
    st.stop()

target = st.selectbox("Select target column", categorical_cols)
y = LabelEncoder().fit_transform(df_model[target])

x = df_model.drop(columns=[target])
x = x.select_dtypes(include=np.number)

if x.empty:
    st.error("No numeric features available")
    st.stop()

# ---- Model-specific preprocessing ----
if nb_type == "Gaussian Naive Bayes":
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = GaussianNB()

elif nb_type == "Multinomial Naive Bayes":
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    model = MultinomialNB()

else:  # Bernoulli Naive Bayes
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    x = (x > 0.5).astype(int)
    model = BernoulliNB()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"Accuracy: {acc:.2f}")
log(f"{nb_type} trained | Accuracy = {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
st.pyplot(fig)
