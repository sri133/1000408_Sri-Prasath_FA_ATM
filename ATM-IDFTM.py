# =====================================================
# ULTRA PREMIUM ATM INTELLIGENCE DASHBOARD (FINAL)
# SAME FEATURES + ADVANCED NEON BLUE UI
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="ATM Intelligence Ultra", layout="wide")

# -----------------------------------------------------
# 🔥 ULTRA PREMIUM NEON BLUE UI (LIKE 2ND APP)
# -----------------------------------------------------
st.markdown("""
<style>

/* BACKGROUND ANIMATION */
.stApp {
    background: radial-gradient(circle at 20% 20%, #001f3f, #000000 80%);
    overflow: hidden;
}

/* GLOWING ANIMATED MESH */
.stApp::before {
    content: "";
    position: fixed;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,150,255,0.15) 10%, transparent 40%),
                radial-gradient(circle, rgba(0,80,255,0.15) 20%, transparent 50%),
                radial-gradient(circle, rgba(0,200,255,0.1) 10%, transparent 40%);
    animation: moveGlow 20s linear infinite;
    z-index: 0;
}

@keyframes moveGlow {
    0% { transform: translate(0,0); }
    50% { transform: translate(-25%, -25%); }
    100% { transform: translate(0,0); }
}

/* GLASS CARDS */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(0,150,255,0.3);
    box-shadow: 0 0 25px rgba(0,150,255,0.4);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000814, #001d3d);
    border-right: 1px solid rgba(0,150,255,0.4);
}

/* TEXT */
h1, h2, h3, h4 {
    color: #e6f1ff;
    text-shadow: 0 0 10px rgba(0,150,255,0.6);
}

/* METRIC CARDS */
div[data-testid="metric-container"] {
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(0,150,255,0.4);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,150,255,0.3);
}

/* BUTTON */
button {
    background: linear-gradient(45deg, #0077ff, #00c6ff);
    color: white;
    border-radius: 10px;
    border: none;
    box-shadow: 0 0 10px rgba(0,150,255,0.7);
}

</style>
""", unsafe_allow_html=True)

st.title("🏧 ATM Intelligence Ultra Dashboard")

# -----------------------------------------------------
# LOAD DATA (UNCHANGED)
# -----------------------------------------------------
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/sri133/1000408_Sri-Prasath_FA_ATM/main/atm_cash_management_dataset.csv"
        return pd.read_csv(url)
    except:
        return pd.read_csv("atm_cash_management_dataset.csv")


df = load_data()

# -----------------------------------------------------
# PREPROCESSING (UNCHANGED + ADDED FEATURES)
# -----------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df.fillna(method="ffill", inplace=True)

df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

df["Is_Weekend"] = df["Date"].dt.day_name().isin(["Saturday","Sunday"]).astype(int)

df = df.sort_values(["ATM_ID","Date"])

df["Rolling_Mean_Withdrawals"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(lambda x: x.rolling(7,1).mean())
df["Daily_Change_Pct"] = df.groupby("ATM_ID")["Total_Withdrawals"].pct_change().fillna(0)*100

# -----------------------------------------------------
# SIDEBAR FILTERS (FULLY KEPT)
# -----------------------------------------------------
st.sidebar.header("Filters")

locations = st.sidebar.multiselect("Location Type", df["Location_Type"].unique(), default=df["Location_Type"].unique())

# -----------------------------------------------------
# FILTERING
# -----------------------------------------------------
filtered_df = df[df["Location_Type"].isin(locations)].copy()

# -----------------------------------------------------
# KPI
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total ATMs", df["ATM_ID"].nunique())
col2.metric("Avg Withdrawals", round(filtered_df["Total_Withdrawals"].mean(),2))
col3.metric("Records", len(filtered_df))

# -----------------------------------------------------
# CHARTS
# -----------------------------------------------------
st.subheader("Withdrawals Trend")
st.plotly_chart(px.line(filtered_df, x="Date", y="Total_Withdrawals"))

# -----------------------------------------------------
# CLUSTERING + PCA
# -----------------------------------------------------
st.subheader("Clustering")

features = ["Total_Withdrawals","Total_Deposits"]

scaler = StandardScaler()
X = scaler.fit_transform(filtered_df[features])

kmeans = KMeans(n_clusters=3)
filtered_df["Cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
comp = pca.fit_transform(X)
filtered_df["PCA1"] = comp[:,0]
filtered_df["PCA2"] = comp[:,1]

st.plotly_chart(px.scatter(filtered_df,x="PCA1",y="PCA2",color="Cluster"))

# -----------------------------------------------------
# ANOMALY (IQR + IF)
# -----------------------------------------------------
Q1 = filtered_df["Total_Withdrawals"].quantile(0.25)
Q3 = filtered_df["Total_Withdrawals"].quantile(0.75)
IQR = Q3 - Q1

filtered_df["IQR_Anomaly"] = (
    (filtered_df["Total_Withdrawals"] < Q1 - 1.5*IQR) |
    (filtered_df["Total_Withdrawals"] > Q3 + 1.5*IQR)
)

iso = IsolationForest(contamination=0.05)
filtered_df["IF_Anomaly"] = iso.fit_predict(X)

st.plotly_chart(px.scatter(filtered_df,x="Date",y="Total_Withdrawals",color="IF_Anomaly"))

# -----------------------------------------------------
# FORECASTING
# -----------------------------------------------------
features = ["Total_Withdrawals","Total_Deposits","Month"]

if len(filtered_df) > 10:
    X = filtered_df[features]
    y = filtered_df["Cash_Demand_Next_Day"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    model = MLPRegressor(max_iter=300)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    st.metric("MAE", round(mean_absolute_error(y_test, pred),2))
    st.metric("R2", round(r2_score(y_test, pred),2))

# -----------------------------------------------------
st.success("Ultra Premium Blue UI Applied Successfully")
