# =====================================================
# UPGRADED ATM INTELLIGENCE DASHBOARD (FINAL)
# Combines App 1 + App 2 Features (No Gemini AI)
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
# UI CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="ATM Intelligence Pro", layout="wide")

# -----------------------------------------------------
# PREMIUM UI
# -----------------------------------------------------
st.markdown("""
<style>
.stApp {background-color:#0b0c10; color:white;}
.block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🏧 ATM Intelligence Pro Dashboard")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("atm_cash_management_dataset.csv")


df = load_data()
df["Date"] = pd.to_datetime(df["Date"])

# -----------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------
df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day_of_Week"] = df["Date"].dt.day_name()
df["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)

df = df.sort_values(["ATM_ID", "Date"])
df["Rolling_Mean_Withdrawals"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(lambda x: x.rolling(7,1).mean())
df["Daily_Change_Pct"] = df.groupby("ATM_ID")["Total_Withdrawals"].pct_change().fillna(0)*100

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
st.sidebar.header("Controls")

locations = st.sidebar.multiselect("Location", df["Location_Type"].unique(), default=df["Location_Type"].unique())

selected_metrics = st.sidebar.multiselect("Select Features", 
    ["Total_Withdrawals","Total_Deposits","Nearby_Competitor_ATMs","Rolling_Mean_Withdrawals"],
    default=["Total_Withdrawals","Total_Deposits"])

k_clusters = st.sidebar.slider("Clusters",2,8,3)
anomaly_rate = st.sidebar.slider("Anomaly Rate",0.01,0.15,0.05)

# -----------------------------------------------------
# FILTER
# -----------------------------------------------------
filtered_df = df[df["Location_Type"].isin(locations)]

# -----------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------
csv = filtered_df.to_csv(index=False).encode()
st.sidebar.download_button("Download Data", csv, "filtered.csv")

# -----------------------------------------------------
# KPI
# -----------------------------------------------------
col1,col2,col3 = st.columns(3)
col1.metric("ATMs", filtered_df["ATM_ID"].nunique())
col2.metric("Avg Withdrawals", round(filtered_df["Total_Withdrawals"].mean(),2))
col3.metric("Records", len(filtered_df))

# =====================================================
# EDA
# =====================================================
st.header("EDA")
st.plotly_chart(px.line(filtered_df,x="Date",y="Total_Withdrawals"))
st.plotly_chart(px.histogram(filtered_df,x="Total_Withdrawals"))

# =====================================================
# CLUSTERING + PCA
# =====================================================
st.header("Clustering")

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
# ANOMALY DETECTION
# =====================================================
st.header("Anomaly Detection")

iso = IsolationForest(contamination=anomaly_rate)
filtered_df["Anomaly"] = iso.fit_predict(X)

st.plotly_chart(px.scatter(filtered_df,x="Date",y="Total_Withdrawals",color="Anomaly"))

st.subheader("Critical Anomalies")
st.dataframe(filtered_df[filtered_df["Anomaly"]==-1])

# =====================================================
# FORECASTING (ORIGINAL MODEL KEPT)
# =====================================================
st.header("Forecasting")

features = ["Total_Withdrawals","Total_Deposits","Month","Week_Number"]

X = filtered_df[features]
y = filtered_df["Cash_Demand_Next_Day"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)

model = MLPRegressor(max_iter=300)
model.fit(X_train,y_train)

pred = model.predict(X_test)

colA,colB = st.columns(2)
colA.metric("MAE", round(mean_absolute_error(y_test,pred),2))
colB.metric("R2", round(r2_score(y_test,pred),2))

st.plotly_chart(px.scatter(x=y_test,y=pred,labels={"x":"Actual","y":"Pred"}))

# =====================================================
# ADVANCED FORECAST ANALYSIS
# =====================================================
st.header("Advanced Forecast Insights")

st.plotly_chart(px.bar(filtered_df.groupby("Day_of_Week")["Total_Withdrawals"].mean().reset_index(),
                       x="Day_of_Week",y="Total_Withdrawals"))

st.plotly_chart(px.line(filtered_df.groupby("Date")["Rolling_Mean_Withdrawals"].mean().reset_index(),
                        x="Date",y="Rolling_Mean_Withdrawals"))

st.plotly_chart(px.histogram(filtered_df,x="Daily_Change_Pct"))

# =====================================================
st.success("App Upgraded Successfully 🚀")
