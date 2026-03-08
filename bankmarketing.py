import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Bank ML Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.big-title {
    font-size:45px;
    font-weight:bold;
    color:#154360;
}
.kpi-card {
    background-color:#F2F4F4;
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🏦 Bank Marketing ML Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Interactive Machine Learning Analytics System")
st.markdown("---")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank.csv")

df = load_data()
df.columns = df.columns.str.strip()

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["📊 Dashboard", "🤖 Model Training", "🔮 Prediction"])

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------
if menu == "📊 Dashboard":

    st.subheader("📌 Dataset Summary")

    yes_count = df["deposit"].value_counts().get(1, 0)
    no_count = df["deposit"].value_counts().get(0, 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Subscribed (Yes)", yes_count)
    col4.metric("Not Subscribed (No)", no_count)

    st.markdown("---")

    # Raw Data Section
    with st.expander("📋 View Raw Dataset"):
        st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # Filters
    st.subheader("🎛 Filter Data")

    age_filter = st.slider("Select Age Range", int(df.age.min()), int(df.age.max()), (20, 60))
    filtered_df = df[(df.age >= age_filter[0]) & (df.age <= age_filter[1])]

    st.write(f"Filtered Data Count: {filtered_df.shape[0]}")

    # Charts Section
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Age Distribution")
        fig, ax = plt.subplots()
        filtered_df["age"].hist(bins=20, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("💰 Balance Distribution")
        fig, ax = plt.subplots()
        filtered_df["balance"].hist(bins=20, ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("🎯 Target Distribution")
    fig, ax = plt.subplots()
    filtered_df["deposit"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------
elif menu == "🤖 Model Training":

    st.subheader("⚙ Model Selection")

    model_option = st.selectbox(
        "Choose Model",
        ["Decision Tree", "Random Forest", "XGBoost", "Logistic Regression"]
    )

    df_model = df.copy()

    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col])

    X = df_model.drop("deposit", axis=1)
    y = df_model["deposit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Model"):

        with st.spinner("Training Model..."):

            if model_option == "Decision Tree":
                model = DecisionTreeClassifier()

            elif model_option == "Random Forest":
                model = RandomForestClassifier(n_estimators=100)

            elif model_option == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            joblib.dump(model, "model.pkl")

        st.success("Model Trained Successfully!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test,y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test,y_pred):.2f}")
        col3.metric("Recall", f"{recall_score(y_test,y_pred):.2f}")
        col4.metric("F1 Score", f"{f1_score(y_test,y_pred):.2f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    # ------------------------------------------------
    #OVERALL FEATURE IMPORTANCE (Random Forest Based)
    # ------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Overall Feature Importance (Random Forest Based)")

    # Encode dataset again
    df_overall = df.copy()
    for col in df_overall.select_dtypes(include="object").columns:
        df_overall[col] = LabelEncoder().fit_transform(df_overall[col])

    X_overall = df_overall.drop("deposit", axis=1)
    y_overall = df_overall["deposit"]

    rf_overall = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_overall.fit(X_overall, y_overall)

    importances = rf_overall.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X_overall.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(9,6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax
    )

    ax.set_title("Overall Feature Importance")
    st.pyplot(fig)
    # ------------------------------------------------
    # MODEL COMPARISON SECTION
    # ------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")

    # Encode dataset
    df_compare = df.copy()
    for col in df_compare.select_dtypes(include="object").columns:
        df_compare[col] = LabelEncoder().fit_transform(df_compare[col])

    X = df_compare.drop("deposit", axis=1)
    y = df_compare["deposit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train All Models
    models = {
        "LR": LogisticRegression(max_iter=1000),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_estimators=100),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    model_names = []

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        model_names.append(name)
        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))
        recall_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))

    # Create 2x2 Layout
    col1, col2 = st.columns(2)

    # F1 Score
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.bar(model_names, f1_list)
        ax1.set_title("F1 Score Comparison")
        ax1.set_ylabel("F1 Score")
        st.pyplot(fig1)

    # Recall
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(model_names, recall_list)
        ax2.set_title("Recall Comparison")
        ax2.set_ylabel("Recall")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    # Precision
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.bar(model_names, precision_list)
        ax3.set_title("Precision Comparison")
        ax3.set_ylabel("Precision")
        st.pyplot(fig3)

    # Accuracy
    with col4:
        fig4, ax4 = plt.subplots()
        ax4.bar(model_names, accuracy_list)
        ax4.set_title("Accuracy Comparison")
        ax4.set_ylabel("Accuracy")
        st.pyplot(fig4)
# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
elif menu == "🔮 Prediction":

    st.subheader("🔍 Enter Customer Details")

    try:
        model = joblib.load("model.pkl")

        input_data = {}
        cols = st.columns(2)
        i = 0

        for col in df.drop("deposit", axis=1).columns:
            if df[col].dtype == "object":
                input_data[col] = cols[i % 2].selectbox(col, df[col].unique())
            else:
                input_data[col] = cols[i % 2].number_input(col, value=0)
            i += 1

        input_df = pd.DataFrame([input_data])

        # Encode again
        for col in input_df.select_dtypes(include="object").columns:
            input_df[col] = LabelEncoder().fit(df[col]).transform(input_df[col])

        if st.button("🔮 Predict Now"):

            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("### 🎯 Prediction Result")

            if prediction[0] == 1:
                st.success("Customer WILL Subscribe ✅")
            else:
                st.error("Customer will NOT Subscribe ❌")

            st.progress(float(probability))
            st.write(f"Subscription Probability: {round(probability*100,2)}%")

    except:
        st.warning("Please train the model first.")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.markdown("✨ Developed using Streamlit | ML Classification Dashboard")
