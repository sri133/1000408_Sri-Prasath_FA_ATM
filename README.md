**ATM Intelligence Ultra Pro 

An interactive Python-based dashboard for monitoring, analyzing, and forecasting ATM cash demand. The dashboard provides actionable insights for banking operations teams, treasury planners, and branch managers to optimize cash management, detect anomalies, and make data-driven decisions.

1. Project Overview

The ATM Intelligence Ultra Pro transforms raw ATM transaction data into interactive visualizations, clusters, and forecasts. It enables users to explore trends over time, identify high- and low-demand ATMs, detect unusual withdrawals, and simulate “what-if” scenarios using demand multipliers.

The application combines data preprocessing, exploratory data analysis, clustering, anomaly detection, and predictive modeling into a single, easy-to-use interface.

2. Dataset

Source: https://github.com/sri133/1000408_Sri-Prasath_FA_ATM/blob/main/atm_cash_management_dataset.csv

Key Features:

ATM & Time Information: ATM_ID, Date, Day_of_Week, Time_of_Day
Transactions: Total_Withdrawals, Total_Deposits, Previous_Day_Cash_Level, Cash_Demand_Next_Day
Contextual Factors: Location_Type, Holiday_Flag, Special_Event_Flag, Weather_Condition, Nearby_Competitor_ATMs

3. Data Preprocessing

Converted Date into datetime format and extracted features: Month, Week_Number, Is_Weekend.

Handled missing values using forward-fill.

Encoded categorical variables numerically:

Day_of_Week, Time_of_Day, Location_Type, Holiday_Flag, Special_Event_Flag, Weather_Condition, Nearby_Competitor_ATMs
Normalized numeric features using StandardScaler and MinMaxScaler.

Checked logical consistency to ensure withdrawals did not exceed available balances.

4. Key Features

4.1 Interactive Filters

Filter data by: Location_Type, Day_of_Week, Time_of_Day, Holiday_Flag, Special_Event_Flag, Weather_Condition, Nearby_Competitor_ATMs.

Demand multiplier slider (1x–3x) for scenario analysis.

Download filtered dataset in CSV format.

4.2 Exploratory Data Analysis

Distribution Analysis: Histograms and box plots for withdrawals and deposits.

Time-Based Trends: Line charts for withdrawal patterns over time.

Day/Time Comparison: Bar plots for Day_of_Week and Time_of_Day.

Holiday & Event Impact: Bar charts for withdrawals on holidays and special events.

External Factors: Box plots for weather conditions and competitor ATMs.

Relationship Analysis: Scatter plot between Previous_Day_Cash_Level and Cash_Demand_Next_Day.

Correlation Analysis: Heatmap for all numeric features.

4.3 Clustering Analysis

Group ATMs based on demand behavior using K-Means Clustering.

Features include: Total_Withdrawals, Total_Deposits, Location_Type, Nearby_Competitor_ATMs.

PCA visualization for 2D cluster representation.

Clusters interpreted as:

High-demand (Urban/Festival ATMs)

Steady-demand (Business hubs)

Low-demand (Rural/Residential ATMs)

4.4 Anomaly Detection

Detect unusual withdrawal patterns using Isolation Forest.

Anomalies highlighted on scatter plots.

Special focus on holidays and special events for early detection of spikes.

4.5 Forecasting

Predict next-day cash demand using MLP Regressor.

Key features: Total_Withdrawals, Total_Deposits, Month.

Performance metrics: MAE and R² Score displayed.

5. Visualizations

Interactive, responsive charts using Plotly Express:

Line, bar, scatter, histogram, and box plots

Rolling averages and smoothed trends

Correlation heatmaps

Ultra-premium UI with gradient backgrounds, styled metrics, and dynamic layouts.

6. Technologies & Libraries

Python

Streamlit – interactive web app interface

Pandas & NumPy – data manipulation and preprocessing

Plotly Express – interactive visualizations

Scikit-learn – clustering, anomaly detection, scaling, and regression



8. Outcome & Insights

Classified ATMs into high, steady, and low-demand groups.

Detected unusual spikes in withdrawals during holidays and special events.

Predicted next-day cash demand for better cash allocation.

Empowered bank managers with interactive, actionable insights for operational decisions.

9. Credits: 

Student Name: Sri Prasath. P

Mentor: Arul Jothi

Course: Data Mining

School Name: Jain Vidyalaya IB World School

This README is fully self-contained, professional, and explains: dataset, preprocessing, interactive features, analysis, and forecasting—all without referencing FA-1 or FA-2.
