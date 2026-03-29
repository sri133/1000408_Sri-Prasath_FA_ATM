# =====================================================
# FINAL ULTRA UI + ALL FEATURES PRESERVED (CORRECTED)
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

st.set_page_config(page_title="ATM Intelligence Ultra", layout="wide")

# ================= UI (NEON BLUE FULL) =================
st.markdown("""
<style>
.stApp {
 background: radial-gradient(circle at 20% 20%, #001f3f, #000000 80%);
}

.stApp::before {
 content: "";
 position: fixed;
 width: 200%; height: 200%;
 background: radial-gradient(circle, rgba(0,150,255,0.15), transparent 40%);
 animation: moveGlow 18s infinite linear;
}
@keyframes moveGlow {
 0% {transform: translate(0,0);} 
 50% {transform: translate(-20%,-20%);} 
 100% {transform: translate(0,0);} 
}

section[data-testid="stSidebar"] {
 background: linear-gradient(#000814,#001d3d);
}

h1,h2,h3 {color:white; text-shadow:0 0 10px #00aaff;}

div[data-testid="metric-container"] {
 background: rgba(0,0,0,0.6);
 border:1px solid #00aaff;
 border-radius:12px;
 box-shadow:0 0 15px #00aaff;
}
</style>
""", unsafe_allow_html=True)

st.title("🏧 ATM Intelligence Dashboard (Final Ultra)")

# ================= LOAD =================
@st.cache_data
def load():
 return pd.read_csv("https://raw.githubusercontent.com/sri133/1000408_Sri-Prasath_FA_ATM/main/atm_cash_management_dataset.csv")

df = load()
df["Date"] = pd.to_datetime(df["Date"])

# ================= ORIGINAL FEATURES =================
df["Month"] = df["Date"].dt.month
df["Week_Number"] = df["Date"].dt.isocalendar().week.astype(int)

# ================= NEW FEATURES =================
df["Is_Weekend"] = df["Date"].dt.day_name().isin(["Saturday","Sunday"]).astype(int)
df = df.sort_values(["ATM_ID","Date"])
df["Rolling_Mean_Withdrawals"] = df.groupby("ATM_ID")["Total_Withdrawals"].transform(lambda x: x.rolling(7,1).mean())
df["Daily_Change_Pct"] = df.groupby("ATM_ID")["Total_Withdrawals"].pct_change().fillna(0)*100

# ================= FILTERS (FULL RESTORED) =================
st.sidebar.header("Filters")

loc = st.sidebar.multiselect("Location", df["Location_Type"].unique(), default=df["Location_Type"].unique())
day = st.sidebar.multiselect("Day", df["Day_of_Week"].unique(), default=df["Day_of_Week"].unique())
time = st.sidebar.multiselect("Time", df["Time_of_Day"].unique(), default=df["Time_of_Day"].unique())
holiday = st.sidebar.multiselect("Holiday", df["Holiday_Flag"].unique(), default=df["Holiday_Flag"].unique())
weather = st.sidebar.multiselect("Weather", df["Weather_Condition"].unique(), default=df["Weather_Condition"].unique())

filtered = df[
 (df["Location_Type"].isin(loc)) &
 (df["Day_of_Week"].isin(day)) &
 (df["Time_of_Day"].isin(time)) &
 (df["Holiday_Flag"].isin(holiday)) &
 (df["Weather_Condition"].isin(weather))
]

# ================= KPI =================
col1,col2,col3 = st.columns(3)
col1.metric("ATMs", df["ATM_ID"].nunique())
col2.metric("Avg Withdrawals", round(filtered["Total_Withdrawals"].mean(),2))
col3.metric("Records", len(filtered))

# ================= ALL ORIGINAL GRAPHS RESTORED =================
st.header("EDA")
st.plotly_chart(px.line(filtered, x="Date", y="Total_Withdrawals"))
st.plotly_chart(px.histogram(filtered, x="Total_Withdrawals"))
st.plotly_chart(px.box(filtered, y="Total_Withdrawals"))
st.plotly_chart(px.scatter(filtered, x="Total_Withdrawals", y="Total_Deposits"))

# ================= CLUSTER =================
features = ["Total_Withdrawals","Total_Deposits"]
scaler = StandardScaler()
X = scaler.fit_transform(filtered[features])

kmeans = KMeans(n_clusters=3)
filtered["Cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
comp = pca.fit_transform(X)
filtered["PCA1"] = comp[:,0]
filtered["PCA2"] = comp[:,1]

st.plotly_chart(px.scatter(filtered,x="PCA1",y="PCA2",color="Cluster"))

# ================= ANOMALY =================
Q1 = filtered["Total_Withdrawals"].quantile(0.25)
Q3 = filtered["Total_Withdrawals"].quantile(0.75)
IQR = Q3 - Q1
filtered["IQR"] = ((filtered["Total_Withdrawals"]<Q1-1.5*IQR)|(filtered["Total_Withdrawals"]>Q3+1.5*IQR))

iso = IsolationForest(contamination=0.05)
filtered["IF"] = iso.fit_predict(X)

st.plotly_chart(px.scatter(filtered,x="Date",y="Total_Withdrawals",color="IF"))

# ================= FORECAST =================
features = ["Total_Withdrawals","Total_Deposits","Month"]

if len(filtered)>10:
 X = filtered[features]
 y = filtered["Cash_Demand_Next_Day"]

 scaler = MinMaxScaler()
 X_scaled = scaler.fit_transform(X)

 X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)

 model = MLPRegressor(max_iter=300)
 model.fit(X_train,y_train)

 pred = model.predict(X_test)

 st.metric("MAE", round(mean_absolute_error(y_test,pred),2))
 st.metric("R2", round(r2_score(y_test,pred),2))

# ================= EXTRA INSIGHTS =================
st.header("Advanced Insights")
st.plotly_chart(px.bar(filtered.groupby("Day_of_Week")["Total_Withdrawals"].mean().reset_index(),x="Day_of_Week",y="Total_Withdrawals"))
st.plotly_chart(px.line(filtered.groupby("Date")["Rolling_Mean_Withdrawals"].mean().reset_index(),x="Date",y="Rolling_Mean_Withdrawals"))
st.plotly_chart(px.histogram(filtered,x="Daily_Change_Pct"))

st.success("Now EVERYTHING is preserved + Ultra UI applied correctly")
