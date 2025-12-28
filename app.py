import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 Random Forest Classifier – Car Evaluation Dataset")
st.markdown(
    """
    This interactive application trains a **Random Forest Classifier** on the  
    **Car Evaluation dataset**, evaluates its performance, and visualizes results.
    """
)

st.divider()

st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test size (%)",
    min_value=10,
    max_value=50,
    value=30,
    step=5
) / 100

n_estimators = st.sidebar.slider(
    "Number of Trees (n_estimators)",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    value=42,
    step=1
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_path = "car_evaluation.csv"
data = load_data(data_path)

st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("**Shape of dataset:**", data.shape)
    st.write("**Columns:**", data.columns.tolist())

with col2:
    st.write("**Sample data:**")
    st.dataframe(data.head(), use_container_width=True)

encoded_data = data.copy()
for col in encoded_data.select_dtypes(include=["object"]).columns:
    encoded_data[col] = encoded_data[col].astype("category").cat.codes

X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

rf_classifier = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state
)
rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")


st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy:.2f}")
m2.metric("Precision", f"{precision:.2f}")
m3.metric("Recall", f"{recall:.2f}")
m4.metric("F1 Score", f"{f1:.2f}")


st.subheader("🧮 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

st.divider()
st.markdown(
    "<center>🚀 Built By Enigjes</center>",
    unsafe_allow_html=True
)
