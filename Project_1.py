import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Page Config with Dark Mode Preference
st.set_page_config(page_title="E-Commerce Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: #ffffff; }
    .stMetric { background-color: #1a1f2c; padding: 20px; border-radius: 12px; border: 1px solid #2d3748; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stMetric div[data-testid="stMetricValue"] { color: #6366f1; font-size: 2rem; font-weight: 700; }
    .prediction-box { background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%); padding: 25px; border-radius: 16px; border: 1px solid #4c1d95; text-align: center; margin-top: 20px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: -0.5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 Executive E-Commerce Performance Dashboard")
st.markdown("Real-time sales analytics, return tracking, and predictive forecasting.")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('ecommerce_data.csv')

df = load_data()

# KPI Calculations
total_sales = df[df['Returned'] == 0]['Sales'].sum()
total_profit = df[df['Returned'] == 0]['Profit'].sum()
return_rate = (df['Returned'].sum() / len(df)) * 100

# KPIs Grid
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Net Revenue (Excl. Returns)", value=f"${total_sales:,.2f}")
with col2:
    st.metric(label="Gross Profit", value=f"${total_profit:,.2f}")
with col3:
    st.metric(label="Return Rate", value=f"{return_rate:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Charts Section
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📦 Top Selling Products by Volume")
    prod_sales = df[df['Returned'] == 0].groupby('Product')['Quantity'].sum().reset_index()
    fig_prod = px.bar(prod_sales, x='Product', y='Quantity', color='Quantity', 
                      color_continuous_scale='Purples', template='plotly_dark')
    fig_prod.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_prod, use_container_width=True)

with col_right:
    st.subheader("📈 Monthly Profit Margins")
    monthly_profit = df[df['Returned'] == 0].groupby('Month')['Profit'].sum().reset_index()
    fig_mon = px.line(monthly_profit, x='Month', y='Profit', markers=True, template='plotly_dark')
    fig_mon.update_traces(line_color='#10b981', line_width=3, marker=dict(size=8, color='#059669'))
    fig_mon.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_mon, use_container_width=True)

st.markdown("---")

# ML Forecasting
st.subheader("🔮 Predictive Analytics")
monthly_sales = df[df['Returned'] == 0].groupby('Month')['Sales'].sum().reset_index()
X = monthly_sales['Month'].values.reshape(-1, 1)
y = monthly_sales['Sales'].values

model = LinearRegression()
model.fit(X, y)
next_month = np.array([[13]])
predicted_sales = model.predict(next_month)[0]

st.markdown(f"""
<div class="prediction-box">
    <span style="color: #a78bfa; text-transform: uppercase; font-size: 0.85rem; font-weight: bold; letter-spacing: 1px;">Machine Learning Forecast</span>
    <h2 style="color: #ffffff; margin: 10px 0 5px 0;">Predicted Revenue for Next Month (Month 13)</h2>
    <h1 style="color: #34d399; margin: 0; font-size: 3rem;">${predicted_sales:,.2f}</h1>
</div>
""", unsafe_allow_html=True)