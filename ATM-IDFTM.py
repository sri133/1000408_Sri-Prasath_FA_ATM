# =====================================================
# 🔥 FINAL PERFECT ATM INTELLIGENCE DASHBOARD (NO FEATURE LOSS)
# ALL FEATURES + ALL FILTERS + ALL GRAPHS + PREMIUM UI
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
st.set_page_config(page_title="ATM Intelligence Ultra Pro", layout="wide")

# -----------------------------------------------------
# 🔥 ULTRA PREMIUM UI
# -----------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #001f3f, #000000 80%);
}

h1,h2,h3 {
    color:#e6f1ff;
    text-shadow:0 0 10px rgba(0,150,255,0.7);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(#000814,#001d3d);
}

div[data-testid="metric-container"] {
    background: rgba(0,0,0,0.6);
    border-radius:15px;
    padding:15px;
    border:1px solid rgba(0,150,255,0.4);
}

button {
    background: linear-gradient(45deg,#0077ff,#00c6ff);
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("🏧 ATM Intelligence Ultra Pro Dashboard")

# -----------------------------------------------------
# LOAD DATA
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
# PREPROCESSING (FULL SAFE)
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
# RAW DATA
# -----------------------------------------------------
st.subheader("📂 Raw Data")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# -----------------------------------------------------
# SIDEBAR FILTERS (ALL RESTORED)
# -----------------------------------------------------
st.sidebar.header("🔍 Advanced Filters")

locations = st.sidebar.multiselect("Location", df["Location_Type"].unique(), df["Location_Type"].unique())
days = st.sidebar.multiselect("Day", df["Day_of_Week"].unique(), df["Day_of_Week"].unique())
times = st.sidebar.multiselect("Time", df["Time_of_Day"].unique(), df["Time_of_Day"].unique())
holidays = st.sidebar.multiselect("Holiday", df["Holiday_Flag"].unique(), df["Holiday_Flag"].unique())
events = st.sidebar.multiselect("Events", df["Special_Event_Flag"].unique(), df["Special_Event_Flag"].unique())
weather = st.sidebar.multiselect("Weather", df["Weather_Condition"].unique(), df["Weather_Condition"].unique())
competitor = st.sidebar.multiselect("Competitor", df["Nearby_Competitor_ATMs"].unique(), df["Nearby_Competitor_ATMs"].unique())

multiplier = st.sidebar.slider("Demand Multiplier (1x - 3x)",1,3,1)

# =====================================================
# ⚙️ ML CONTROLS (EXTRA FILTERS)
# =====================================================
st.sidebar.markdown("## ⚙️ ML Controls")

selected_metrics = st.sidebar.multiselect(
    "Select Features",
    df.select_dtypes(include=np.number).columns.tolist(),
    default=["Total_Withdrawals", "Total_Deposits"]
)

k_clusters = st.sidebar.slider("Clusters", 2, 8, 3)
anomaly_rate = st.sidebar.slider("Anomaly Rate", 0.01, 0.15, 0.05)
# -----------------------------------------------------
# FILTER DATA
# -----------------------------------------------------
filtered_df = df[
    (df["Location_Type"].isin(locations)) &
    (df["Day_of_Week"].isin(days)) &
    (df["Time_of_Day"].isin(times)) &
    (df["Holiday_Flag"].isin(holidays)) &
    (df["Special_Event_Flag"].isin(events)) &
    (df["Weather_Condition"].isin(weather)) &
    (df["Nearby_Competitor_ATMs"].isin(competitor))
].copy()

filtered_df["Adjusted_Withdrawals"] = filtered_df["Total_Withdrawals"] * multiplier

# -----------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------
st.sidebar.download_button("Download Data", filtered_df.to_csv(index=False), "atm_data.csv")

# -----------------------------------------------------
# KPI
# -----------------------------------------------------
col1,col2,col3 = st.columns(3)
col1.metric("ATMs", df["ATM_ID"].nunique())
col2.metric("Avg Withdrawals", round(filtered_df["Adjusted_Withdrawals"].mean(),2))
col3.metric("Records", len(filtered_df))

# -----------------------------------------------------
# ALL GRAPHS RESTORED
# -----------------------------------------------------
st.header("📊 Analysis")

st.plotly_chart(px.line(filtered_df,x="Date",y="Adjusted_Withdrawals"))
st.plotly_chart(px.histogram(filtered_df,x="Adjusted_Withdrawals"))
st.plotly_chart(px.box(filtered_df,y="Adjusted_Withdrawals"))
st.plotly_chart(px.bar(filtered_df.groupby("Day_of_Week")["Adjusted_Withdrawals"].mean().reset_index(),x="Day_of_Week",y="Adjusted_Withdrawals"))
st.plotly_chart(px.line(filtered_df.groupby("Date")["Rolling_Mean_Withdrawals"].mean().reset_index(),x="Date",y="Rolling_Mean_Withdrawals"))


# =====================================================
# 📊 CORRELATION HEATMAP (8th GRAPH)
# =====================================================
st.subheader("📊 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 1:
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)
# -----------------------------------------------------
# CLUSTERING
# -----------------------------------------------------
st.header("📍 Clustering")
features = selected_metrics
scaler = StandardScaler()
X = scaler.fit_transform(filtered_df[features])

kmeans = KMeans(n_clusters=k_clusters)
filtered_df["Cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
comp = pca.fit_transform(X)
filtered_df["PCA1"] = comp[:,0]
filtered_df["PCA2"] = comp[:,1]

st.plotly_chart(px.scatter(filtered_df,x="PCA1",y="PCA2",color="Cluster"))

# -----------------------------------------------------
# ANOMALY
# -----------------------------------------------------
st.header("⚠️ Anomaly")
iso = IsolationForest(contamination=anomaly_rate)
filtered_df["Anomaly"] = iso.fit_predict(X)

st.plotly_chart(px.scatter(filtered_df,x="Date",y="Adjusted_Withdrawals",color="Anomaly"))

# -----------------------------------------------------
# FORECAST
# -----------------------------------------------------
st.header("🤖 Forecast")
features = ["Total_Withdrawals","Total_Deposits","Month"]

if len(filtered_df)>10:
    X = filtered_df[features]
    y = filtered_df["Cash_Demand_Next_Day"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)

    model = MLPRegressor(max_iter=300)
    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    st.metric("MAE",round(mean_absolute_error(y_test,pred),2))
    st.metric("R2",round(r2_score(y_test,pred),2))

# -----------------------------------------------------
st.success("✅ PERFECT: ALL FEATURES + ALL FILTERS + PREMIUM UI")
