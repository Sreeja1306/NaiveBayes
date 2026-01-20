import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Page Config ----------------
st.set_page_config("Naive Bayes Classifier", layout="wide")
st.title("Naive Bayes Classification – End-to-End Streamlit App")

# ---------------- Sidebar ----------------
st.sidebar.header("Model Configuration")

nb_type = st.sidebar.selectbox(
    "Select Naive Bayes Variant",
    ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"]
)

test_size = st.sidebar.slider("Test Size", 0.2, 0.4, 0.25)

# ---------------- Step 1: Upload Dataset ----------------
st.header("Step 1: Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully")

# ---------------- Step 2: Dataset Preview ----------------
st.header("Step 2: Dataset Preview")
st.dataframe(df.head())
st.write("Shape:", df.shape)
st.write("Missing Values:")
st.write(df.isnull().sum())

# ---------------- Step 3: Select Target Column ----------------
st.header("Step 3: Select Target Column")

categorical_cols = df.select_dtypes(include=["object"]).columns

if len(categorical_cols) == 0:
    st.error("Dataset must contain at least one categorical target column")
    st.stop()

target_col = st.selectbox("Choose target column", categorical_cols)

# ---------------- Step 4: Preprocessing ----------------
st.header("Step 4: Data Preprocessing")

# Handle missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Encode target
y = LabelEncoder().fit_transform(df[target_col])

# Select numeric features
X = df.drop(columns=[target_col])
X = X.select_dtypes(include=np.number)

if X.empty:
    st.error("No numeric feature columns found")
    st.stop()

st.success("Preprocessing completed")

# ---------------- Step 5: Scaling & Model Selection ----------------
st.header("Step 5: Model Training")

if nb_type == "Gaussian Naive Bayes":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianNB()

elif nb_type == "Multinomial Naive Bayes":
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = MultinomialNB()

else:  # Bernoulli NB
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = (X_scaled > 0.5).astype(int)
    model = BernoulliNB()

# ---------------- Step 6: Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42
)

model.fit(X_train, y_train)

# ---------------- Step 7: Evaluation ----------------
st.header("Step 6: Model Evaluation")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.success(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# ---------------- Step 8: Prediction Probabilities ----------------
st.header("Step 7: Prediction Probabilities")

proba_df = pd.DataFrame(
    y_proba,
    columns=[f"Class {c}" for c in model.classes_]
)

st.dataframe(proba_df.head())

st.info(
    "Naive Bayes computes posterior probabilities for each class "
    "and predicts the class with the highest probability."
)
