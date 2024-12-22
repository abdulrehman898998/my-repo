import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("ML Model Evaluation with Robust Preprocessing")

# Function to handle imbalanced data
def handle_imbalanced_data(X, y, min_samples_per_class=2):
    class_counts = Counter(y)
    if min(class_counts.values()) < min_samples_per_class:
        st.warning(f"Warning: Some classes have fewer than {min_samples_per_class} samples. Using random oversampling instead of SMOTE.")
        unique_classes = np.unique(y)
        max_samples = max(class_counts.values())
        X_resampled = []
        y_resampled = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            resampled_indices = np.random.choice(cls_indices, size=max_samples, replace=True)
            X_resampled.append(X[resampled_indices])
            y_resampled.extend([cls] * max_samples)

        return np.vstack(X_resampled), np.array(y_resampled)
    else:
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Dataset Preview:")
        st.dataframe(df)

        # Handle missing values
        if st.checkbox("Handle missing values"):
            imputer = SimpleImputer(strategy="most_frequent")
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            st.write("Missing values handled. Updated dataset:")
            st.dataframe(df)

        # Feature selection
        features = st.multiselect("Select features", options=df.columns)
        target = st.selectbox("Select target variable", options=df.columns)

        if features and target:
            # Encode categorical data
            st.write("Encoding categorical data...")
            for col in df.columns:
                if df[col].dtype == "object":
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    st.write(f"Encoded column: {col}")

            X = df[features].values
            y = df[target].values

            # Display class distribution before balancing
            if st.checkbox("Show class distribution"):
                st.write("Original class distribution:")
                st.write(pd.Series(y).value_counts())

            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Handle imbalanced data (for classification only)
                if len(np.unique(y)) > 2 and st.checkbox("Handle imbalanced data"):
                    X_train, y_train = handle_imbalanced_data(X_train, y_train)

                st.write("Train-Test Split Completed:")
                st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

                # Select task type
                task = st.radio("Select task type", options=["Classification", "Regression"])

                if task == "Classification":
                    model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Decision Tree"])
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression(random_state=42)
                    elif model_type == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)

                    # Train model
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Metrics
                    st.write("Classification Report:")
                    report = classification_report(y_test, predictions, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, predictions)
                    st.write("Confusion Matrix:")
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
                    st.pyplot(plt)

                elif task == "Regression":
                    model_type = st.selectbox("Select Model", ["Random Forest", "Linear Regression", "Decision Tree"])
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(random_state=42)
                    elif model_type == "Linear Regression":
                        model = LinearRegression()
                    elif model_type == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=42)

                    # Train model
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Metrics
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, predictions)
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"Root Mean Squared Error: {rmse:.4f}")
                    st.write(f"R-squared: {r2:.4f}")

                    # Visualization
                    st.write("Actual vs Predicted:")
                    plt.figure(figsize=(8, 6))
                    plt.scatter(y_test, predictions, alpha=0.5)
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    st.pyplot(plt)

            except ValueError as ve:
                st.error(f"Error during data splitting: {ve}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Error details:", str(e))

