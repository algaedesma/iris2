import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# Load model
with open("naive_bayes_iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load iris data for reference
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
df_iris['species'] = df_iris['target'].apply(lambda x: iris.target_names[x])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Description", "🔮 Predict", "ℹ️ About Model"])

# Page 1: Data Description
if page == "📊 Data Description":
    st.title("📊 Iris Dataset Overview")
    st.write("This page shows a description of the classic Iris dataset.")
    st.write(df_iris.head())
    st.write("Class distribution:")
    st.bar_chart(df_iris['species'].value_counts())

# Page 2: Prediction
elif page == "🔮 Predict":
    st.title("🔮 Predict Iris Species")
    st.write("Enter values below to predict the Iris species.")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_species = iris.target_names[prediction[0]]

    st.success(f"The predicted species is: **{predicted_species.capitalize()}**")

# Page 3: About Model
elif page == "ℹ️ About Model":
    st.title("ℹ️ About This Model")
    st.write("""
        This is a simple machine learning deployment using **Naive Bayes** trained on the Iris dataset.

        - Model type: GaussianNB
        - Training data: 150 Iris flower samples
        - Features: Sepal & Petal dimensions
        - Classes: Setosa, Versicolor, Virginica

        This demo was built with ❤️ using **Streamlit**.
    """)
