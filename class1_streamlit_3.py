# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from io import StringIO

# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---

@st.cache_data
def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
        df_cleaned = df.copy()
        if 'customerID' in df_cleaned.columns:
            df_cleaned = df_cleaned.drop('customerID', axis=1)
        if 'TotalCharges' in df_cleaned.columns:
            df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')
            total_charges_median = df_cleaned['TotalCharges'].median()
            df_cleaned['TotalCharges'] = df_cleaned['TotalCharges'].fillna(total_charges_median)
        else:
            st.warning("Column 'TotalCharges' not found. Some features might not work as expected.")
        if 'Churn' in df_cleaned.columns:
            if df_cleaned['Churn'].dtype == 'object':
                 df_cleaned['Churn'] = df_cleaned['Churn'].map({'No': 0, 'Yes': 1}).fillna(-1)
                 if (df_cleaned['Churn'] == -1).any():
                     st.warning("Some 'Churn' values were not 'Yes' or 'No' and have been handled/removed. Please check your target column.")
                     df_cleaned = df_cleaned[df_cleaned['Churn'] != -1]
            elif not pd.api.types.is_numeric_dtype(df_cleaned['Churn']):
                st.error("The 'Churn' column is not in a recognizable format (Yes/No or numeric 0/1).")
                return None
        else:
            st.error("Target column 'Churn' not found in the uploaded dataset.")
            return None
        return df_cleaned
    return None

@st.cache_data
def preprocess_data(_df_cleaned):
    if _df_cleaned is None or 'Churn' not in _df_cleaned.columns:
        return None, None, None, None, None, None
    X = _df_cleaned.drop('Churn', axis=1)
    y = _df_cleaned['Churn']
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names_out = None
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names_out

def get_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, max_depth=5)
    elif model_name == "SVC":
        return SVC(random_state=42, probability=True)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
    elif model_name == "MLP Classifier":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=10)
    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model_name == "Naive Bayes":
        return GaussianNB()
    return None

def get_voting_model(selected_models):
    base_estimators = []
    for name in selected_models:
        if name == "Voting Classifier":
            continue
        model = get_model(name)
        if model is not None:
            base_estimators.append((name[:3], model))
    if len(base_estimators) >= 2:
        return VotingClassifier(estimators=base_estimators, voting='hard')
    else:
        return None

def get_param_grid(model_name):
    if model_name == "Logistic Regression":
        return {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    elif model_name == "Decision Tree":
        return {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
    elif model_name == "SVC":
        return  [
            {'kernel': ['linear'], 'C': [0.1, 1, 10]},
            {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
        ]
    elif model_name == "Random Forest":
        return {'n_estimators': [50, 100], 'max_depth': [5, 10], 'max_features': ['sqrt', 'log2']}
    elif model_name == "Gradient Boosting":
        return {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    elif model_name == "MLP Classifier":
        return {'hidden_layer_sizes': [(64,), (64, 32)], 'alpha': [0.0001, 0.001], 'learning_rate_init': [0.001, 0.01]}
    elif model_name == "KNN":
        return {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    elif model_name == "Naive Bayes":
        return {}  # GaussianNB usually has no hyperparameters to tune
    return None



# --- Streamlit App UI ---
st.title("Customer Churn Prediction App") # Emoji removed
st.markdown("""
Welcome to the Customer Churn Predictor!
This app allows you to upload your customer data (in CSV format), select machine learning models,
and predict customer churn. It's designed for datasets similar to the Telco Customer Churn dataset.
""")

# --- Sidebar ---
st.sidebar.header("Settings") # Emoji removed
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

model_options = [
    "Logistic Regression", "Decision Tree", "SVC",
    "Random Forest", "Gradient Boosting", "MLP Classifier","KNN","Naive Bayes","Voting Classifier"
]

st.sidebar.subheader("Hyperparameter Tuning")
enable_tuning = st.sidebar.checkbox("Enable Grid Search Tuning", value=False)


if uploaded_file:
    df_cleaned = load_and_clean_data(uploaded_file)
    if df_cleaned is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Selection")
        selected_models = st.sidebar.multiselect(
            "Choose models to train:", model_options, default=[model_options[0]]
        )
        run_button = st.sidebar.button("Train Selected Models", type="primary", use_container_width=True)
        st.sidebar.markdown("---")
        
        st.header("Data Overview & EDA") # Emoji removed
        st.write("First 5 rows of the cleaned dataset:")
        st.dataframe(df_cleaned.head())
        st.write("Basic statistics of numerical features:")
        st.dataframe(df_cleaned.describe())
        if 'Churn' in df_cleaned.columns:
            st.subheader("Churn Distribution")
            churn_counts = df_cleaned['Churn'].value_counts()
            churn_labels_map = {0: 'No Churn', 1: 'Churn'}
            churn_names = [churn_labels_map.get(idx, f'Unknown ({idx})') for idx in churn_counts.index]
            fig_churn_pie = px.pie(values=churn_counts.values,
                                   names=churn_names,
                                   title='Customer Churn Distribution', hole=0.3)
            fig_churn_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_churn_pie, use_container_width=True)

        if run_button and selected_models:
            st.header("Model Training & Evaluation") # Emoji removed
            with st.spinner("Preprocessing data..."):
                preprocess_result = preprocess_data(df_cleaned)
            if preprocess_result is None or preprocess_result[0] is None:
                st.error("Data preprocessing failed. Please check your dataset and console for errors.")
            else:
                X_train_p, X_test_p, y_train, y_test, preprocessor, feature_names = preprocess_result
                if X_train_p is not None and X_test_p is not None:
                    st.success("Data preprocessed successfully!")
                    st.write(f"Training data shape: {X_train_p.shape}, Test data shape: {X_test_p.shape}")
                    for model_name in selected_models:
                        st.subheader(f"Results for: {model_name}")
                        if model_name == "Voting Classifier":
                            model_instance = get_voting_model(selected_models)
                            if model_instance is None:
                                st.warning("Voting Classifier requires at least two base models to work.")
                                continue
                        else:
                            model_instance = get_model(model_name)
                        if model_instance is not None:
                            with st.spinner(f"Training {model_name}..."):
                                try:
                                    if enable_tuning:
                                        st.info(f"Tuning hyperparameters for {model_name} using GridSearchCV...")
                                        param_grid = get_param_grid(model_name)
                                        if param_grid:
                                            grid_search = GridSearchCV(model_instance, param_grid, cv=3,
                                                                       scoring='accuracy', n_jobs=-1)
                                            grid_search.fit(X_train_p, y_train)
                                            best_model = grid_search.best_estimator_
                                            st.success(f"Best parameters for {model_name}: {grid_search.best_params_}")
                                        else:
                                            best_model = model_instance.fit(X_train_p,
                                                                            y_train)  # fallback for models like Naive Bayes
                                            st.info(f"{model_name} has no tunable parameters or tuning skipped.")
                                    else:
                                        best_model = model_instance.fit(X_train_p, y_train)

                                    y_pred = best_model.predict(X_test_p)
                                    acc = accuracy_score(y_test, y_pred)
                                    st.write(f"**Accuracy:** {acc*100:.2f}%")
                                    cm = confusion_matrix(y_test, y_pred)
                                    cm_labels = ['No Churn', 'Churn']
                                    if cm.shape == (2,2):
                                        fig_cm = ff.create_annotated_heatmap(
                                            z=cm, x=cm_labels, y=cm_labels, colorscale='Blues',
                                            showscale=True
                                        )
                                        fig_cm.update_layout(
                                            title=f'Confusion Matrix for {model_name}',
                                            xaxis_title="Predicted Label",
                                            yaxis_title="Actual Label"
                                        )
                                        st.plotly_chart(fig_cm, use_container_width=True)
                                    else:
                                        st.warning(f"Could not generate standard 2x2 confusion matrix for {model_name}. Matrix shape: {cm.shape}. Raw matrix: \n{cm}")
                                except AttributeError as e:
                                    st.error(f"An AttributeError occurred while training or predicting with {model_name}: {e}")
                                    st.error("This might indicate an issue with the model's state before or during fitting. Ensure the data is preprocessed correctly.")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred with {model_name}: {e}")
                        else:
                            st.error(f"Could not retrieve/initialize model: {model_name}")
                        st.markdown("---")
                else:
                    st.error("Processed training or testing data is missing. Cannot proceed with model training.")
        elif run_button and not selected_models:
            st.warning("Please select at least one model to train.")
else:
    st.info("Upload a CSV file using the sidebar to get started!") # Emoji removed

st.sidebar.markdown("---")
st.sidebar.markdown("Created for Applied ML Class")