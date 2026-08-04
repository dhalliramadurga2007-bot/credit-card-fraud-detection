import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #1e3a8a);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.85;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    text-align: center;
    border: 1px solid #e5e7eb;
}

.card-title {
    color: #6b7280;
    font-size: 14px;
}

.card-value {
    color: #111827;
    font-size: 28px;
    font-weight: 700;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.safe {
    background-color: #ecfdf5;
    border: 2px solid #10b981;
}

.danger {
    background-color: #fef2f2;
    border: 2px solid #ef4444;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
}

.result-value {
    font-size: 38px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()


# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")


df = load_data()


# =========================
# SIDEBAR
# =========================
st.sidebar.title("💳 FraudShield AI")
st.sidebar.caption("Credit Card Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Fraud Detection",
        "📊 Analytics",
        "🤖 Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI-powered system designed to identify "
    "potentially fraudulent credit card transactions."
)


# ==========================================================
# DASHBOARD
# ==========================================================
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>💳 FraudShield AI</h1>
        <p>Intelligent Credit Card Fraud Detection System</p>
        <p>Machine Learning powered transaction risk analysis</p>
    </div>
    """, unsafe_allow_html=True)

    total_transactions = len(df)
    fraud_transactions = int(df["Class"].sum())
    legitimate_transactions = total_transactions - fraud_transactions
    fraud_rate = (fraud_transactions / total_transactions) * 100

    st.subheader("📈 Transaction Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Total Transactions</div>
            <div class="card-value">{total_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Fraud Transactions</div>
            <div class="card-value">{fraud_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Legitimate Transactions</div>
            <div class="card-value">{legitimate_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Fraud Rate</div>
            <div class="card-value">{fraud_rate:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Transaction Distribution")

        counts = df["Class"].value_counts()

        chart_df = pd.DataFrame({
            "Transaction Type": [
                "Legitimate",
                "Fraud"
            ],
            "Count": [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        })

        st.bar_chart(
            chart_df.set_index("Transaction Type")
        )

    with col2:

        st.subheader("💰 Transaction Amount")

        st.line_chart(
            df["Amount"].head(1000)
        )

    st.divider()

    st.subheader("🚀 How the System Works")

    a, b, c = st.columns(3)

    with a:
        st.info(
            "**1️⃣ Input**\n\n"
            "Transaction information is selected."
        )

    with b:
        st.warning(
            "**2️⃣ AI Analysis**\n\n"
            "Machine Learning model analyzes transaction patterns."
        )

    with c:
        st.success(
            "**3️⃣ Result**\n\n"
            "System predicts Fraud or Legitimate."
        )


# ==========================================================
# FRAUD DETECTION
# ==========================================================
elif page == "🔍 Fraud Detection":

    st.markdown("""
    <div class="hero">
        <h1>🔍 Fraud Detection</h1>
        <p>
        Test real transactions from the dataset
        using the trained Machine Learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🧪 Test a Real Transaction")

    test_type = st.radio(
        "Choose transaction type",
        [
            "Random Transaction",
            "Fraud Transaction",
            "Legitimate Transaction"
        ],
        horizontal=True
    )

    if test_type == "Fraud Transaction":

        available = df[df["Class"] == 1]

    elif test_type == "Legitimate Transaction":

        available = df[df["Class"] == 0]

    else:

        available = df

    if st.button(
        "🎲 Select Transaction",
        use_container_width=True
    ):

        selected = available.sample(1).iloc[0]

        st.session_state["selected_transaction"] = selected


    if "selected_transaction" in st.session_state:

        selected = st.session_state["selected_transaction"]

        st.divider()

        st.subheader("💳 Selected Transaction")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Transaction Amount",
                f"${selected['Amount']:.2f}"
            )

        with col2:

            st.metric(
                "Transaction Time",
                f"{selected['Time']:.0f}"
            )

        with col3:

            if selected["Class"] == 1:

                st.metric(
                    "Actual Status",
                    "🚨 FRAUD"
                )

            else:

                st.metric(
                    "Actual Status",
                    "✅ LEGITIMATE"
                )


        st.write("")

        if st.button(
            "🤖 RUN AI PREDICTION",
            use_container_width=True
        ):

            feature_columns = [
                "Time",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "V10",
                "V11",
                "V12",
                "V13",
                "V14",
                "V15",
                "V16",
                "V17",
                "V18",
                "V19",
                "V20",
                "V21",
                "V22",
                "V23",
                "V24",
                "V25",
                "V26",
                "V27",
                "V28",
                "Amount"
            ]

            transaction = pd.DataFrame(
                [
                    [
                        selected[col]
                        for col in feature_columns
                    ]
                ],
                columns=feature_columns
            )

            # Scale Time and Amount
            transaction[["Time", "Amount"]] = scaler.transform(
                transaction[["Time", "Amount"]]
            )

            # Prediction
            prediction = model.predict(transaction)[0]

            probability = model.predict_proba(transaction)[0][1]

            st.divider()

            st.subheader("🤖 AI Prediction")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:

                    st.error(
                        "🚨 FRAUD DETECTED"
                    )

                else:

                    st.success(
                        "✅ LEGITIMATE TRANSACTION"
                    )

            with col2:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(
                float(probability),
                text=f"Fraud Risk: {probability * 100:.2f}%"
            )

            st.divider()

            actual = int(selected["Class"])

            if prediction == actual:

                st.success(
                    "✅ CORRECT PREDICTION — "
                    "The AI prediction matches "
                    "the actual dataset label."
                )

            else:

                st.error(
                    "❌ INCORRECT PREDICTION — "
                    "The AI prediction does not match "
                    "the actual dataset label."
                )

            st.subheader("📋 Transaction Features")

            display_data = transaction.copy()

            display_data["Actual Class"] = actual

            display_data["Predicted Class"] = prediction

            st.dataframe(
                display_data,
                use_container_width=True
            )

    else:

        st.info(
            "👆 Choose a transaction type and click "
            "'Select Transaction' to begin testing."
        )


# ==========================================================
# ANALYTICS
# ==========================================================
elif page == "📊 Analytics":

    st.markdown("""
    <div class="hero">
        <h1>📊 Analytics</h1>
        <p>
        Explore patterns and statistics
        from the transaction dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Transaction Class Distribution")

        class_counts = df["Class"].value_counts()

        st.bar_chart(class_counts)

    with col2:

        st.subheader("Transaction Amount Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            df["Amount"],
            bins=50
        )

        ax.set_xlabel("Transaction Amount")

        ax.set_ylabel(
            "Number of Transactions"
        )

        ax.set_title(
            "Transaction Amount Distribution"
        )

        st.pyplot(fig)

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ==========================================================
# MODEL INFORMATION
# ==========================================================
elif page == "🤖 Model Information":

    st.markdown("""
    <div class="hero">
        <h1>🤖 Machine Learning Model</h1>
        <p>
        Information about the fraud detection model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    with col2:

        st.metric(
            "Training Rows",
            f"{int(len(df) * 0.8):,}"
        )

    with col3:

        st.metric(
            "Testing Rows",
            f"{int(len(df) * 0.2):,}"
        )

    st.divider()

    st.subheader("📌 Model Details")

    st.write("""
    **Algorithm:** Logistic Regression

    **Problem Type:** Binary Classification

    **Target Variable:** Class

    **Class 0:** Legitimate Transaction

    **Class 1:** Fraudulent Transaction

    **Preprocessing:** StandardScaler

    **Train-Test Split:** 80% Training / 20% Testing

    **Class Imbalance Handling:** Balanced class weights
    """)

    st.subheader("🎯 Model Evaluation")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Fraud Recall",
            "Fraud Precision",
            "Fraud F1-Score"
        ],
        "Score": [
            "97.55%",
            "91.84%",
            "6.09%",
            "10.99%"
        ]
    })

    st.table(metrics)

    st.warning(
        "Because fraud transactions are extremely rare, "
        "accuracy alone should not be used to judge the model."
    )

    st.success(
        "The model prioritizes detecting fraudulent transactions "
        "by using balanced class weights."
    )