# =====================================================
# ATM INTELLIGENCE DEMAND FORECASTING (FINAL FULL APP)
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
st.set_page_config(page_title="ATM Intelligence Dashboard", layout="wide")
st.title("🏧 ATM Intelligence Demand Forecasting Dashboard")

# -----------------------------------------------------
# LOAD DATA (FROM 1ST CODE)
# -----------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/sri133/1000408_SriPrasath.P_AIY1_FA2-ATM/main/atm_cash_management_dataset.csv"
    return pd.read_csv(url)

df = load_data()

# -----------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

df = df.sort_values(["ATM_ID","Date"])

df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day_Name"] = df["Date"].dt.day_name()
df["Is_Weekend"] = df["Day_Name"].isin(["Saturday","Sunday"]).astype(int)

df["Rolling_Mean_Withdrawals"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(lambda x: x.rolling(7,1).mean())
df["Daily_Change_Pct"] = df.groupby("ATM_ID")["Total_Withdrawals"].pct_change().fillna(0)*100

# Encode categorical safely
cat_cols = ["Day_of_Week","Time_of_Day","Location_Type","Weather_Condition"]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.header("⚙️ Controls")

locations = st.sidebar.multiselect(
    "Location",
    df["Location_Type"].unique(),
    default=df["Location_Type"].unique()
)

k_clusters = st.sidebar.slider("Clusters (K)",2,8,3)
anomaly_rate = st.sidebar.slider("Anomaly %",0.01,0.15,0.05)

selected_metrics = st.sidebar.multiselect(
    "Metrics",
    ["Total_Withdrawals","Total_Deposits","Previous_Day_Cash_Level",
     "Nearby_Competitor_ATMs","Rolling_Mean_Withdrawals"],
    default=["Total_Withdrawals","Total_Deposits"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df["Date"].min(), df["Date"].max()]
)

filtered_df = df[
    (df["Location_Type"].isin(locations)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy()

# -----------------------------------------------------
# KPI
# -----------------------------------------------------
col1,col2,col3 = st.columns(3)
col1.metric("Total ATMs", df["ATM_ID"].nunique())
col2.metric("Avg Withdrawal", round(filtered_df["Total_Withdrawals"].mean(),2))
col3.metric("Records", len(filtered_df))

# -----------------------------------------------------
# TABS
# -----------------------------------------------------
tabs = st.tabs([
    "🌐 Overview",
    "📊 EDA",
    "📈 Advanced Analytics",
    "📍 Clustering",
    "⚠️ Anomaly Detection",
    "🤖 Forecasting"
])

# =====================================================
# 🌐 OVERVIEW
# =====================================================
with tabs[0]:
    st.subheader("📈 Withdrawals Over Time")
    st.plotly_chart(px.line(filtered_df,x="Date",y="Total_Withdrawals"))

    st.dataframe(filtered_df.head(200))

# =====================================================
# 📊 EDA
# =====================================================
with tabs[1]:
    st.plotly_chart(px.histogram(filtered_df,x="Total_Withdrawals"))
    st.plotly_chart(px.bar(filtered_df,x="Day_of_Week",y="Total_Withdrawals"))
    st.plotly_chart(px.box(filtered_df,x="Weather_Condition",y="Total_Withdrawals"))

    corr = filtered_df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# 📈 ADVANCED ANALYTICS
# =====================================================
with tabs[2]:
    if len(selected_metrics) < 2:
        st.warning("⚠️ Select at least 2 metrics")
    else:
        corr_matrix = filtered_df[selected_metrics].corr()
        st.plotly_chart(px.imshow(corr_matrix, text_auto=True))

        st.plotly_chart(px.box(
            filtered_df,
            y=selected_metrics[0],
            color="Location_Type"
        ))

# =====================================================
# 📍 CLUSTERING
# =====================================================
with tabs[3]:
    if len(selected_metrics) < 2:
        st.warning("⚠️ Select at least 2 metrics for clustering")
    else:
        X = filtered_df[selected_metrics]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

        if len(selected_metrics) > 3:
            pca = PCA(n_components=3)
            comp = pca.fit_transform(X_scaled)
            filtered_df["PCA1"] = comp[:,0]
            filtered_df["PCA2"] = comp[:,1]
            filtered_df["PCA3"] = comp[:,2]

            st.plotly_chart(px.scatter_3d(
                filtered_df,
                x="PCA1", y="PCA2", z="PCA3",
                color="Cluster"
            ))
        else:
            st.plotly_chart(px.scatter(
                filtered_df,
                x=selected_metrics[0],
                y=selected_metrics[1],
                color="Cluster"
            ))

# =====================================================
# ⚠️ ANOMALY DETECTION
# =====================================================
with tabs[4]:

    # IQR
    Q1 = filtered_df["Total_Withdrawals"].quantile(0.25)
    Q3 = filtered_df["Total_Withdrawals"].quantile(0.75)
    IQR = Q3 - Q1

    filtered_df["IQR_Anomaly"] = (
        (filtered_df["Total_Withdrawals"] < Q1 - 1.5*IQR) |
        (filtered_df["Total_Withdrawals"] > Q3 + 1.5*IQR)
    )

    st.subheader("IQR Method")
    st.plotly_chart(px.scatter(filtered_df,
                               x="Date",
                               y="Total_Withdrawals",
                               color="IQR_Anomaly"))

    # Isolation Forest
    if len(selected_metrics) >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(filtered_df[selected_metrics])

        iso = IsolationForest(contamination=anomaly_rate, random_state=42)
        filtered_df["IF_Anomaly"] = iso.fit_predict(X_scaled)

        st.subheader("Isolation Forest")
        st.plotly_chart(px.scatter(filtered_df,
                                   x="Date",
                                   y="Total_Withdrawals",
                                   color="IF_Anomaly"))

# =====================================================
# 🤖 FORECASTING
# =====================================================
with tabs[5]:

    st.subheader("📊 Smart Insights")

    if len(filtered_df)>0:
        top_day = filtered_df.groupby("Day_of_Week")["Total_Withdrawals"].mean().idxmax()
        top_time = filtered_df.groupby("Time_of_Day")["Total_Withdrawals"].mean().idxmax()

        st.write(f"🔥 Peak Day: {top_day}")
        st.write(f"⏰ Peak Time: {top_time}")

    st.subheader("🤖 ML Forecast")

    features = ["Total_Withdrawals","Total_Deposits","Location_Type","Month","Week_Number"]

    if len(filtered_df)>10:
        X = filtered_df[features]
        y = filtered_df["Cash_Demand_Next_Day"]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

        model = MLPRegressor(max_iter=500,random_state=42)
        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        colA,colB = st.columns(2)
        colA.metric("MAE", round(mean_absolute_error(y_test,pred),2))
        colB.metric("R²", round(r2_score(y_test,pred),3))

        st.plotly_chart(px.scatter(x=y_test,y=pred))

    st.subheader("📈 Forecast Analytics")

    st.plotly_chart(px.line(filtered_df,
                            x="Date",
                            y="Rolling_Mean_Withdrawals"))

    st.plotly_chart(px.histogram(filtered_df,
                                 x="Daily_Change_Pct"))

# =====================================================
st.success("✅ FINAL APP RUNNING PERFECTLY")
