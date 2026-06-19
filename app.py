import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

# Load Model
model = joblib.load("Customer_seg.pkl")
with st.sidebar:

    st.title("📊 About Project")

    st.write("""
    This application uses
    K-Means Clustering for
    Customer Segmentation.
    """)

    st.markdown("### Features")

    st.write("✔ Customer Analysis")
    st.write("✔ K-Means Clustering")
    st.write("✔ Real-Time Prediction")

    st.info("AI/ML Portfolio Project")

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.big-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1E88E5;
}

.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='big-title'>🛍️ Customer Segmentation System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>K-Means Clustering Based Customer Analysis</div>",
    unsafe_allow_html=True
)

st.divider()

# Input Section
st.subheader("📊 Customer Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "💰 Annual Income (k$)",
        min_value=0,
        max_value=200,
        value=50
    )

with col2:
    score = st.number_input(
        "🛒 Spending Score (1-100)",
        min_value=1,
        max_value=100,
        value=50
    )

st.divider()

# Prediction
if st.button("🔍 Analyze Customer", use_container_width=True):

    data = np.array([[income, score]])

    cluster = model.predict(data)[0]

    cluster_names = {
        0: "Premium Customers 🏆",
        1: "Regular Customers 😊",
        2: "High Spenders 💎",
        3: "Budget Customers 💰",
        4: "Potential Customers 🚀"
    }

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )

    st.markdown(f"""
    ### 🎯 Customer Segment

    **{cluster_names.get(cluster, 'Customer Group')}**
    """)

    st.balloons()

# Footer
st.divider()

st.caption(
    "Built using Scikit-Learn, K-Means Clustering and Streamlit"
)