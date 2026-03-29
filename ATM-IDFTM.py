# =====================================================
# FINAL UPGRADED ATM INTELLIGENCE DASHBOARD
# (ALL ORIGINAL FEATURES PRESERVED + NEW FEATURES ADDED)
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

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="ATM Intelligence Pro", layout="wide")

# -----------------------------------------------------
# PREMIUM UI (FROM 2ND APP)
# -----------------------------------------------------
st.markdown("""
<style>
body {background-color:#0b0c10;}
.stApp {
    background: radial-gradient(circle at top, #0b0c10, #000000);
    color:#c5c6c7;
}

.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    border:1px solid rgba(102,252,241,0.2);
    box-shadow: 0 0 20px rgba(102,252,241,0.1);
}

h1,h2,h3 {color:white;}

.sidebar .sidebar-content {
    background: linear-gradient(#0b0c10,#1f2833);
}

</style>
""", unsafe_allow_html=True)

st.title("🏧 ATM Intelligence Demand Forecasting Dashboard")

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
# PREPROCESSING (ORIGINAL + NEW FEATURES)
# -----------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# ORIGINAL
df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

# NEW FEATURES
df["Is_Weekend"] = df["Date"].dt.day_name().isin(["Saturday","Sunday"]).astype(int)
df = df.sort_values(["ATM_ID","Date"])
df["Rolling_Mean_Withdrawals"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(lambda x: x.rolling(7,1).mean())
df["Daily_Change_Pct"] = df.groupby("ATM_ID")["Total_Withdrawals"].pct_change().fillna(0)*100

cat_cols = ["Day_of_Week", "Time_of_Day", "Location_Type", "Weather_Condition"]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# -----------------------------------------------------
# RAW DATA VIEW (KEPT)
# -----------------------------------------------------
st.subheader("📂 Raw Dataset")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# -----------------------------------------------------
# SIDEBAR FILTERS (FULLY PRESERVED)
# -----------------------------------------------------
st.sidebar.header("🔍 Advanced Filters")

locations = st.sidebar.multiselect("Location Type", df["Location_Type"].unique(), default=df["Location_Type"].unique())
days = st.sidebar.multiselect("Day of Week", df["Day_of_Week"].unique(), default=df["Day_of_Week"].unique())
times = st.sidebar.multiselect("Time of Day", df["Time_of_Day"].unique(), default=df["Time_of_Day"].unique())
holidays = st.sidebar.multiselect("Holiday Flag", df["Holiday_Flag"].unique(), default=df["Holiday_Flag"].unique())
events = st.sidebar.multiselect("Special Events", df["Special_Event_Flag"].unique(), default=df["Special_Event_Flag"].unique())
weather = st.sidebar.multiselect("Weather Condition", df["Weather_Condition"].unique(), default=df["Weather_Condition"].unique())
competitor = st.sidebar.multiselect("Nearby Competitor ATMs", df["Nearby_Competitor_ATMs"].unique(), default=df["Nearby_Competitor_ATMs"].unique())

date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])

# NEW CONTROLS
st.sidebar.header("⚙️ ML Controls")
selected_metrics = st.sidebar.multiselect("Select Features", df.select_dtypes(include=np.number).columns.tolist(), default=["Total_Withdrawals","Total_Deposits"])
k_clusters = st.sidebar.slider("Clusters",2,8,3)
anomaly_rate = st.sidebar.slider("Anomaly Rate",0.01,0.15,0.05)

# -----------------------------------------------------
# FILTER DATA (UNCHANGED)
# -----------------------------------------------------
filtered_df = df[
    (df["Location_Type"].isin(locations)) &
    (df["Day_of_Week"].isin(days)) &
    (df["Time_of_Day"].isin(times)) &
    (df["Holiday_Flag"].isin(holidays)) &
    (df["Special_Event_Flag"].isin(events)) &
    (df["Weather_Condition"].isin(weather)) &
    (df["Nearby_Competitor_ATMs"].isin(competitor)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy()

# -----------------------------------------------------
# DOWNLOAD FEATURE (NEW)
# -----------------------------------------------------
st.sidebar.download_button("Download Data", filtered_df.to_csv(index=False), "filtered.csv")

# -----------------------------------------------------
# KPI (UNCHANGED)
# -----------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total ATMs", df["ATM_ID"].nunique())
col2.metric("Avg Withdrawal", round(filtered_df["Total_Withdrawals"].mean(), 2))
col3.metric("Records", len(filtered_df))

# =====================================================
# EDA (UNCHANGED)
# =====================================================
st.header("📊 Exploratory Data Analysis")
st.plotly_chart(px.line(filtered_df, x="Date", y="Total_Withdrawals"))
st.plotly_chart(px.histogram(filtered_df, x="Total_Withdrawals"))

# =====================================================
# CLUSTERING (UPGRADED)
# =====================================================
st.header("📍 ATM Clustering")

if len(selected_metrics) >= 2:
    scaler = StandardScaler()
    X = scaler.fit_transform(filtered_df[selected_metrics])
    kmeans = KMeans(n_clusters=k_clusters)
    filtered_df["Cluster"] = kmeans.fit_predict(X)

    if len(selected_metrics) > 2:
        pca = PCA(n_components=2)
        comp = pca.fit_transform(X)
        filtered_df["PCA1"] = comp[:,0]
        filtered_df["PCA2"] = comp[:,1]
        st.plotly_chart(px.scatter(filtered_df,x="PCA1",y="PCA2",color="Cluster"))
    else:
        st.plotly_chart(px.scatter(filtered_df,x=selected_metrics[0],y=selected_metrics[1],color="Cluster"))

# =====================================================
# ANOMALY DETECTION (BOTH IQR + ISOLATION FOREST)
# =====================================================
st.header("⚠️ Anomaly Detection")

# ORIGINAL IQR
Q1 = filtered_df["Total_Withdrawals"].quantile(0.25)
Q3 = filtered_df["Total_Withdrawals"].quantile(0.75)
IQR = Q3 - Q1
filtered_df["IQR_Anomaly"] = (
    (filtered_df["Total_Withdrawals"] < Q1 - 1.5*IQR) |
    (filtered_df["Total_Withdrawals"] > Q3 + 1.5*IQR)
)

# NEW Isolation Forest
if len(selected_metrics) >= 2:
    iso = IsolationForest(contamination=anomaly_rate)
    filtered_df["IF_Anomaly"] = iso.fit_predict(X)

st.plotly_chart(px.scatter(filtered_df,x="Date",y="Total_Withdrawals",color="IQR_Anomaly"))

st.subheader("Critical Anomalies")
st.dataframe(filtered_df[filtered_df.get("IF_Anomaly",0)==-1])

# =====================================================
# FORECASTING (UNCHANGED)
# =====================================================
st.header("🤖 Demand Forecasting")

test_size = st.slider("Select Test Size", 0.1, 0.4, 0.2)

features = ["Total_Withdrawals","Total_Deposits","Location_Type","Month","Week_Number"]

if len(filtered_df) > 10:
    X = filtered_df[features]
    y = filtered_df["Cash_Demand_Next_Day"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size)

    model = MLPRegressor(max_iter=500)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    colA, colB = st.columns(2)
    colA.metric("MAE", round(mean_absolute_error(y_test, pred), 2))
    colB.metric("R² Score", round(r2_score(y_test, pred), 3))

    st.plotly_chart(px.scatter(x=y_test, y=pred))

# =====================================================
# ADVANCED FORECAST INSIGHTS (NEW)
# =====================================================
st.header("🔮 Advanced Insights")

st.plotly_chart(px.bar(filtered_df.groupby("Day_of_Week")["Total_Withdrawals"].mean().reset_index(), x="Day_of_Week", y="Total_Withdrawals"))
st.plotly_chart(px.line(filtered_df.groupby("Date")["Rolling_Mean_Withdrawals"].mean().reset_index(), x="Date", y="Rolling_Mean_Withdrawals"))
st.plotly_chart(px.histogram(filtered_df, x="Daily_Change_Pct"))

# =====================================================
st.success("✅ Fully Upgraded App (All Original Features + Enhancements)")














