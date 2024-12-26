import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import numpy as np
from streamlit_gsheets import GSheetsConnection

st.set_page_config(layout="wide")

url = "https://docs.google.com/spreadsheets/d/1cZvi2XmgW1NkKwUz2DxbjhWSkNjJfFTg3130N6BO-k8/edit?usp=sharing"

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(spreadsheet=url, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Remove rows where the 'User' column is NaN
df = df.dropna(subset=['User'])

df['Date'] = pd.to_datetime(df['Date'])

# Sidebar filters
users = df['User'].unique()
selected_user = st.sidebar.selectbox('Select User', options=['All'] + list(users))

# Filter by user
if selected_user != 'All':
    df = df[df['User'] == selected_user]

# Sidebar: Filter by Sport
sports = df['Sport'].unique()
selected_sport = st.sidebar.selectbox('Select Sport', options=['All'] + list(sports))

# Filter by sport
if selected_sport != 'All':
    df = df[df['Sport'] == selected_sport]

# Date filter
date_range = st.sidebar.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
else:
    st.sidebar.warning("Please select both start and end dates.")

# Summary stats
w_count = (df['Win_Loss_Push'] == 'w').sum()
l_count = (df['Win_Loss_Push'] == 'l').sum()
p_count = (df['Win_Loss_Push'] == 'p').sum()
total_records = w_count + l_count + p_count  # Total wins, losses, and pushes

win_percentage = (w_count / total_records) * 100 if total_records > 0 else 0
total_units = df['Units_W_L'].sum()

# Display metrics
st.header("Summary Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Wins", w_count)
    
with col2:
    st.metric("Total Losses", l_count)

with col3:
    st.metric("Total Pushes", p_count)

with col4:
    st.metric("Win Percentage", f"{win_percentage:.2f}%")

with col5:
    st.metric("Total Units", f"{total_units:.2f}")

# Cumulative Units Calculation
df_cumulative = df.groupby('Date').agg({'Units_W_L': 'sum'}).cumsum().reset_index()
df_cumulative.rename(columns={'Units_W_L': 'Units'}, inplace=True)

# Create Daily Bar Chart using Plotly
df_daily_sum = df.groupby('Date')['Units_W_L'].sum().reset_index()
fig_daily = go.Figure()

fig_daily.add_trace(go.Bar(
    x=df_daily_sum['Date'],
    y=df_daily_sum['Units_W_L'],
    marker=dict(color=df_daily_sum['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_daily_sum['Units_W_L'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

fig_daily.update_layout(
    title='Daily Units Won / Lost',
    xaxis_title='Date',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Create Weekly Bar Chart using Plotly
df['Week'] = df['Date'].dt.to_period('W').dt.start_time  # Convert dates to start of the week
df_weekly_sum = df.groupby('Week')['Units_W_L'].sum().reset_index()

fig_weekly = go.Figure()

fig_weekly.add_trace(go.Bar(
    x=df_weekly_sum['Week'],
    y=df_weekly_sum['Units_W_L'],
    marker=dict(color=df_weekly_sum['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_weekly_sum['Units_W_L'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

fig_weekly.update_layout(
    title='Weekly Units Won / Lost',
    xaxis_title='Week',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Filter data for side plays
df_filtered = df[df['Side_Play'] == 1]
df_daily_filtered_sum = df_filtered.groupby('Date')['Units_W_L'].sum().reset_index()

# Create Daily Bar Chart for filtered data
fig_daily_filtered = go.Figure()
fig_daily_filtered.add_trace(go.Bar(
    x=df_daily_filtered_sum['Date'],
    y=df_daily_filtered_sum['Units_W_L'],
    marker=dict(color=df_daily_filtered_sum['Units_W_L'].apply(lambda x: 'green' if x > 0 else 'red')),
    text=df_daily_filtered_sum['Units_W_L'].round(2),
    textposition='auto',
    hoverinfo='x+y+text',
))

fig_daily_filtered.update_layout(
    title='Daily Units Won / Lost Side Plays',
    xaxis_title='Date',
    yaxis_title='Units Won / Lost',
    showlegend=False,
    template='plotly_white',
    xaxis_tickangle=-45,
)

# Display the charts
st.plotly_chart(fig_daily, key='daily_chart')
st.plotly_chart(fig_daily_filtered, key='daily_filtered_chart')  # Unique key
st.plotly_chart(fig_weekly, key='weekly_chart')  # Unique key

# Summary table
summary_table = df.groupby('Sport')['Units_W_L'].sum().reset_index()
summary_table.rename(columns={'Units_W_L': 'Units'}, inplace=True)
summary_table['Units'] = summary_table['Units'].round(2)

summary_table = summary_table.sort_values(by='Units', ascending=False)

st.subheader("Units Summary by Sport")
st.table(summary_table)

# Create a DataFrame with User-wise stats (WinPct and Units)
user_stats = df.groupby('User').agg(
    WinPct=('Win_Loss_Push', lambda x: (x == 'w').sum() / len(x) * 100),
    Units=('Units_W_L', 'sum')
).reset_index()

user_stats['WinPct'] = user_stats['WinPct'].round(2)
user_stats['Units'] = user_stats['Units'].round(2)

# Display User-wise stats table
st.subheader("User Win Percentage and Units")
st.dataframe(user_stats)

# Calendar for daily units
calendar_data = df.groupby(df['Date'].dt.date)['Units_W_L'].sum().reset_index()
calendar_data['Date'] = pd.to_datetime(calendar_data['Date'])

df = df.sort_values(by='Date', ascending=False)

# Display full data
st.header('Full Data')
df['Date'] = df['Date'].dt.strftime('%m/%d/%Y')
st.dataframe(df)

# Create the cumulative plot
fig = px.line(df_cumulative, x='Date', y='Units', title='Cumulative Units Over Time')

# Display the plot
st.plotly_chart(fig, key='cumulative_chart')
